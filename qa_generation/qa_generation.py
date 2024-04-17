import traceback
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import json
import time
from openai_config import OpenAIConfig
import os
from tqdm import tqdm

"""
The string formatting function _strip_str and the prompt used to interact 
with the GPT model are adapted from the Gorilla repository 
(https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py).
"""

def _strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by LLM.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]


def _extract_answer_and_reference(answer: str):
    """
    Helper function for extracting the answer and reference from the answer string.
    Used for GPT-4-Turbo.
    """
    begin_ref_tag = "##begin_ref##"
    end_ref_tag = "##end_ref##"
    answer_tag = "<ANSWER>:"

    # Find the positions for reference extraction
    begin_ref = answer.find(begin_ref_tag)
    end_ref = answer.find(end_ref_tag)

    if begin_ref == -1 or end_ref == -1:
        reference = None
    else:
        begin_ref += len(begin_ref_tag)
        reference = answer[begin_ref:end_ref].strip()

    # Find the position for the answer
    answer_start = answer.find(answer_tag)
    if answer_start == -1:
        extracted_answer = None
    else:
        answer_start += len(answer_tag)
        extracted_answer = answer[answer_start:].strip()

    return extracted_answer, reference

def _prep_data(data_dir: str) -> list:
    """
    Helper function for reading a JSONL file and extract data.
    """
    all_data = []

    for filename in os.listdir(data_dir):

        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_dir, filename)

            # Read and parse the JSONL file
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append((data["id"], data["contents"]))

    return all_data

class QAPairsGenerator:
    def __init__(self, num_qa: int = 3, q_model: str = "gpt-3.5-turbo", a_model: str = "gpt-4-turbo"):
        self.num_qa = num_qa
        self.q_model = q_model
        self.a_model = a_model
        self.client = None

    def init_openai_client(self):
        """
        Initialize the OpenAI client.
        """
        # Load the OpenAI API key
        api_key = OpenAIConfig.PROXY_API_KEY
        base_url = OpenAIConfig.PROXY_BASE_URL

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_question(self, chunk: str) -> list[str]:
        """
        Generate questions from the chunk of text.
        """
        respone = self.client.chat.completions.create(
            model = self.q_model,
            messages=[
                {"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'" % (self.num_qa)},
                {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
                {"role": "user", "content": str(chunk)}
            ]
        )

        queries = respone.choices[0].message.content.split("\n")
        queries = [_strip_str(q) for q in queries]
        queries = [q for q in queries if any(c.isalpha() for c in q)]

        return queries 
    
    def generate_answer(self, chunk: str, question: str) -> tuple[str, str]:
        """
        Generate an answer to the question from the chunk of text.
        """
        # Generate the user prompt
        if self.a_model == "gpt-3.5-turbo":
            user_prompt = "Question: {question}\nContext: {context}\nAnswer this question using the information given in the context above, the answer should be succinct. If the provided context does not explicitly contain or imply the answer to the question, you should only answer 'None'.".format(question=question, context=chunk)
        elif self.a_model == "gpt-4-turbo":
            user_prompt = "Question: {question}\nContext: {context}\nAnswer this question using the information given in the context above. Here is things to pay attention to:\n- Please include the sentence from the context you used to generate your answer between ##begin_ref## and ##end_ref##.\n- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinc.\n- If the provided context does not explicitly contain or imply the answer to the question, you should only answer 'None'.".format(question=question, context=chunk)

        # Generate the answer
        response = self.client.chat.completions.create(
            model = self.a_model,
            messages=[
                {"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."},
                {"role": "user", "content": user_prompt}
            ],
            n=1,
            temperature=0
        )

        response_content = response.choices[0].message.content

        if self.a_model == "gpt-3.5-turbo":
            answer, reference = response_content, None
        elif self.a_model == "gpt-4-turbo":
            answer, reference = _extract_answer_and_reference(response_content)

        return answer, reference
    
    def generate_qa_pairs(self, id: str, chunk: str):
        """
        Generate QA pairs from the chunk of text.
        """
        qa_pairs = []

        qs = self.generate_question(chunk)
        
        for i, q in enumerate(qs):
            data = {
                "id": id + "_" + str(i),
                "question": q,
                "doc_chunk": chunk,
                "answer": None,
            }

            answer, _ = self.generate_answer(chunk, q)
            
            # If the answer is not None, then add it to the data
            if answer != None and "None" not in answer:
                data["answer"] = answer
                qa_pairs.append(data)

        return qa_pairs
    
    def generate_qa_pairs_from_chunks(self, chunks: list[str], output_file: str, n_worker: int = 128) -> None:
        """
        Generate QA pairs from the list of chunks of text.
        """
        executor = ThreadPoolExecutor(max_workers=n_worker)

        with open(output_file, 'w', encoding='utf-8') as f: 
            for i in tqdm(range(0, len(chunks), n_worker)):
                # start_time = time.time()
                print(f"Processing batch {i // n_worker} out of {len(chunks) // n_worker}")
                batch = chunks[i: i + n_worker]
                def task(entry):
                    id, chunk = entry
                    try:
                        qa_pairs = self.generate_qa_pairs(id, chunk)
                    except Exception as e:  # just ignore error
                        print(e)
                    for qa_pair in qa_pairs:
                        f.write(json.dumps(qa_pair) + "\n")

                results = executor.map(task, batch)  # submit the job in a detached manner
                for result in results:  # wait for the current batch to finish before next batch
                    pass

                # end_time = time.time()
                # print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    qag = QAPairsGenerator(num_qa=3, q_model="gpt-3.5-turbo", a_model="gpt-4-turbo")
    qag.init_openai_client()
    
    # TODO: Change the corpus_dir and paq_dir to the appropriate directories
    corpus_dir = "corpus/statperls/chunk/Steven"
    paq_dir = "PAQ/statperls/Steven_chunk_paq.jsonl"

    chunks = _prep_data(corpus_dir)

    qag.generate_qa_pairs_from_chunks(chunks, paq_dir)
    