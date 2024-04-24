import argparse
from collections import deque
import jsonlines
from sagemaker.jumpstart.model import JumpStartModel
import sagemaker
import boto3
import json
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import time
import traceback
from pprint import pp


PROMPT_TEMPLATE = \
"""[INST]<<SYS>>
You are a helpful assistant who answers questions on biomedical queries. Please provide an ANSWER to the QUESTION based on the information given in CONTEXT
<</SYS>>
CONTEXT: {context}
QUESTION: {question}
[/INST]
ANSWER:"""


def _build_llama2_prompt(context, question):
    return PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)


class SageMakerLlama27B:

    def __init__(self, sagemaker_arn_role, sagemaker_endpoint_name):
        self.role = sagemaker_arn_role
        self.endpoint_name = sagemaker_endpoint_name
        llama_model = JumpStartModel(model_id="meta-textgeneration-llama-2-7b-f", model_version="2.*", role=self.role)
        print("Deploying Llama-2 7B Chat to SageMaker...")
        llama_model.deploy(initial_instance_count=1,
                           instance_type="ml.g5.2xlarge",
                           endpoint_name=self.endpoint_name,
                           accept_eula=True)

    @classmethod
    def shut_down(cls, endpoint_name):
        """
        Shut down Sagemaker endpoint in destructor in order to save cost
        """
        print(f"Shutting down Sagemaker endpoint {endpoint_name}...")
        sagemaker_session = sagemaker.Session()
        sagemaker_session.delete_endpoint(endpoint_name)
        sagemaker_session.delete_endpoint_config(endpoint_name)

    @classmethod
    def prompt(cls, endpoint_name: str, question: str, context: str, repeats: int, max_new_tokens: int = 256,
               top_p: float = 0.9, temperature: float = 0.6, print_prompt=False) -> list[str]:
        prompt = _build_llama2_prompt(context, question)

        if print_prompt:
            print(prompt)

        payload = {
            "inputs": prompt,

        }

        payload = {
            "inputs": [
                [
                    {"role": "system", "content": "You are a helpful assistant who answers questions on biomedical "
                                                  "queries. Please provide an ANSWER to the QUESTION based on the "
                                                  "information given in CONTEXT. Please just provide a short answer."},
                    {"role": "user", "content": f"CONTEXT: {context}\nQUESTION: {question}\nANSWER:"}
                ]
            ] * repeats,
            "parameters": {"max_new_tokens": max_new_tokens,
                           "top_p": top_p,
                           "temperature": temperature,
                           "return_full_text": False}
        }

        runtime = boto3.client("runtime.sagemaker")
        payload = json.dumps(payload, indent=4).encode("utf-8")
        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType="application/json",
                                           CustomAttributes='accept_eula=true',
                                           Body=payload)
        response_body = json.loads(response["Body"].read())
        generations = [e["generation"]["content"].strip() for e in response_body]
        return generations

def _process_corpus(corpus: list[str]) -> tuple[list[str], list[list[str]]]:
    org_corpus_set = set()
    org_corpus = deque()
    tokenized_corpus = deque()
    for doc in corpus:
        if doc in org_corpus_set:
            continue

        tokenized_doc = _normalize_text(doc)
        tokenized_corpus.append(tokenized_doc)
        org_corpus.append(doc)
        org_corpus_set.add(doc)

    return list(org_corpus), list(tokenized_corpus)


def _normalize_text(doc):
    punctuations = "`~!@#$%^&*()_+[]\\;',./{}|:\"<>?"
    d = doc.lower()
    for punc in punctuations:
        d = d.replace(punc, "")
    tokenized_doc = d.split()
    return tokenized_doc


class Retriever(object):

    def __init__(self):
        docs = []
        with jsonlines.open("knowledge_source/knowledge_source.jsonl", "r") as reader:
            for entry in reader:
                docs.append(entry["text"])

        self.org_corpus, self.tokenized_corpus = _process_corpus(docs)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, question: str):
        tokenized_question = _normalize_text(question)
        return self.bm25.get_top_n(tokenized_question, self.org_corpus, n=1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for simulating a simple RAG system on test questions.")
    parser.add_argument("input_jsonl", help="Path to test JSONL file.")
    parser.add_argument("output_jsonl", help="Path to output test result JSONL file.")
    parser.add_argument("--arn_role", required=False, type=str, help="SageMaker ARN role. This will start a SageMaker endpoint if specified")
    parser.add_argument("--shut_down", action="store_true", help="Shut down SageMaker endpoint.")
    args = parser.parse_args()

    if args.arn_role: SageMakerLlama27B(args.arn_role, "hw4-endpoint")
    pool = ThreadPoolExecutor(max_workers=2)
    retriever = Retriever()
    total_entries, completed_entries = 0, 0

    def _process_one_entry(entry: dict):
        NUM_GENERATIONS = 4
        # tag entry with split info
        if isinstance(entry["id"], str) and "pubmed" in entry["id"]:
            entry["split"] = "pubmed"
        elif isinstance(entry["id"], int):
            entry["split"] = "surreal"
        else:
            entry["split"] = "covered"

        # run retrieval if not covered
        if entry["split"] != "covered":
            entry["doc_chunk"] = retriever.retrieve(entry["question"])

        start = time.time()
        entry["generations"] = SageMakerLlama27B.prompt("hw4-endpoint", entry["question"], entry["doc_chunk"], NUM_GENERATIONS)
        entry["generation_time"] = time.time() - start

        with jsonlines.open(args.output_jsonl, "a") as writer:
            writer.write(entry)

        global completed_entries, total_entries
        completed_entries += 1
        print(f"Processed {completed_entries}/{total_entries}")

    with jsonlines.open(args.output_jsonl, "w") as writer:
        writer.write(None)
    futures = []
    with jsonlines.open(args.input_jsonl, "r") as reader:
        for entry in reader:
            total_entries += 1
            # futures.append(pool.submit(_process_one_entry, entry))
            try:
                _process_one_entry(entry)
            except Exception as e:
                entry_id = entry["id"]
                print(f"entry {entry_id} failed.")
                print(e)
                traceback.print_exc()

        # for future in futures:
        #     future.result()

    if args.shut_down: SageMakerLlama27B.shut_down("hw4-endpoint")
