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
        entry["ground_truth"] = True if tag == "not match" else False
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
            {"role": "system", "content": "You are a helpful assistant classifying ANSWER to a QUESTION that is based "
                                          "on wrong assumptions. Please say 'respond' if ANSWER responds to the wrong "
                                          "assumption, 'doubt' if ANSWER doubts the assumption of the QUESTION,  "
                                          "or 'do not know' if the ANSWER admits absence of knowledge."},
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {generated_answer}"}
        ],
        max_tokens=10
    )
    tag = response.choices[0].message.content.strip().lower().replace(".", "")

    if tag == "respond" or tag == "doubt" or tag == "do not know":
        entry["tag"] = tag
        entry["ground_truth"] = True if tag == "respond" else False
        return entry
    else:
        print(f"GPT provided invalid answer ({tag}) to entry ID {entry_id} (split: pubmed). Writing to error output file.")
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
