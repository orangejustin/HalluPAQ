import argparse
import jsonlines
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

_OPENAI_KEY, _ERROR_FILE = "", ""


def _send_covered_request(entry):
    """
    Send a request OpenAI API to check generated answer for an in-coverage question
    """
    assert entry["split"] == "covered"
    client = OpenAI(api_key=_OPENAI_KEY)
    entry_id, question, true_answer, generated_answer = entry["id"], entry["question"], entry["answer"], entry["generations"][0]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can tell if ANSWER2 provides the same "
                                          "information as ANSWER1. Simply answer 'true' or 'false'."},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER1: {true_answer}\n\nANSWER2: {generated_answer}"}
        ],
        max_tokens=10
    )
    ground_truth = response.choices[0].message.content.strip().lower().replace(".", "")

    if ground_truth == "true" or ground_truth == "false":
        entry["ground_truth"] = True if ground_truth == "false" else False
        return entry
    else:
        print(f"GPT provided invalid answer ({ground_truth}) to entry ID {entry_id} (split: covered). Writing to error output file.")
        with jsonlines.open(_ERROR_FILE, "a") as writer:
            writer.write(entry)


def _send_pubmed_request(entry):
    """
    Send a request OpenAI API to check generated answer for a PubMed question
    """
    assert entry["split"] == "pubmed"
    client = OpenAI(api_key=_OPENAI_KEY)
    entry_id, question, true_answer, generated_answer = entry["id"], entry["question"], entry["answer"], entry["generations"][0]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can tell if ANSWER2 provides the same "
                                          "information as ANSWER1. If ANSWER2 simply admits if absence of knowledge, "
                                          "please answer 'do not know'. Otherwise, If the two answers match, "
                                          "just say 'match'. If two two answers do not match, just say 'not match'."},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER1: {true_answer}\n\nANSWER2: {generated_answer}"}
        ],
        max_tokens=10
    )
    tag = response.choices[0].message.content.strip().lower().replace(".", "")

    if tag == "match" or tag == "not match" or tag == "do not know":
        entry["tag"] = tag
        entry["ground_truth"] = False if tag == "match" else True
        return entry
    else:
        print(f"GPT provided invalid answer ({tag}) to entry ID {entry_id} (split: pubmed). Writing to error output file.")
        with jsonlines.open(_ERROR_FILE, "a") as writer:
            writer.write(entry)


def _send_surreal_request(entry):
    """
    Send a request OpenAI API to check generated answer for a surreal question
    """
    assert entry["split"] == "surreal"
    client = OpenAI(api_key=_OPENAI_KEY)
    entry_id, question, generated_answer = entry["id"], entry["question"], entry["generations"][0]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Given a QUESTION that involves a non-existent concept or unrelated "
                                          "entities, classify the following ANSWER based on its reaction to the "
                                          "question. The classification categories are:\n\n"
                                          "1. The ANSWER is being tricked by the question: This category should be "
                                          "chosen if the answer treats the nonsensical or flawed concept as valid and "
                                          "attempts to provide a serious explanation or context.\n\n"
                                          "2. ANSWER spots a problem in the question: This category should be chosen "
                                          "if the answer identifies that the question is based on a non-existent "
                                          "concept or incorrect assumptions, and clarifies why the question is flawed "
                                          "or makes no sense.\n\nPlease just respond 1 or 2"},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {generated_answer}"}
        ],
        max_tokens=10
    )
    tag = response.choices[0].message.content.strip().lower().replace(".", "").replace("(", "").replace(")", "")

    if tag == "1" or tag == "2":
        entry["tag"] = "tricked" if tag == "1" else "not trickek"
        entry["ground_truth"] = True if tag == "1" else False
        return entry
    else:
        print(f"GPT provided invalid answer ({tag}) to entry ID {entry_id} (split: surreal). Writing to error output file.")
        with jsonlines.open(_ERROR_FILE, "a") as writer:
            writer.write(entry)

def _send_request(entry):
    if entry["split"] == "covered":
        return _send_covered_request(entry)
    elif entry["split"] == "pubmed":
        return _send_pubmed_request(entry)
    elif entry["split"] == "surreal":
        return _send_surreal_request(entry)
    else:
        raise ValueError("Unknown split:", entry["split"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to QA system output file, which contains entries to be tagged.")
    parser.add_argument("output_file", help="Path to tagged output file.")
    parser.add_argument("error_output_file", help="Path to put entries that encountered problem.")
    parser.add_argument("openai_api_key", help="OpenAI API key.")
    parser.add_argument("--n_threads", default=4, type=int, help="Number of threads to send requests")
    args = parser.parse_args()
    _OPENAI_KEY, _ERROR_FILE = args.openai_api_key, args.error_output_file
    pool = ThreadPoolExecutor(max_workers=args.n_threads)

    with jsonlines.open(args.input_file, "r") as reader:
        entries = [entry for entry in reader]
    # clear error output file
    with jsonlines.open(_ERROR_FILE, "w") as writer:
        writer.write_all([])
    # clear output file
    with jsonlines.open(args.output_file, "w") as writer:
        writer.write_all([])

    output_entries = []
    for i in range(0, len(entries), args.n_threads):
        batch_entries = entries[i: i + args.n_threads]
        results = pool.map(_send_request, batch_entries)

        with jsonlines.open(args.output_file, "a") as writer:
            writer.write_all(results)
