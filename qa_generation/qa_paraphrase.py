import jsonlines
from openai_config import OpenAIConfig
from openai import OpenAI
import argparse
from concurrent.futures import ThreadPoolExecutor

_counter = 0

def paraphrase_questions(input_file, output_file, api_key, n_thread: int = 10):
    global _counter
    _counter = 0
    client = OpenAI(api_key=api_key)
    exec = ThreadPoolExecutor(max_workers=n_thread)

    # Open the input and output files
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        results = []

        for entry in reader:

            def task(jsonl_entry):
                global _counter
                original_question = jsonl_entry.get('question')
                if original_question:
                    try:
                        # Request the OpenAI API to paraphrase the question
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant who can paraphrase a sentence given to you. Please paraphrase this question such that it is different from the original."},
                                {"role": "user", "content": original_question}
                            ],
                            max_tokens=100
                        )
                        paraphrased_question = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"Error processing question: {original_question}")
                        print(e)
                        paraphrased_question = original_question  # Use original if error occurs
                else:
                    paraphrased_question = ""

                # Update the entry with the paraphrased question
                jsonl_entry['question'] = paraphrased_question
                jsonl_entry['covered'] = True
                writer.write(jsonl_entry)
                print(f"Line {_counter} written to output file.")
                _counter += 1

            results.append(exec.submit(task, entry))

        for res in results:
            res.result()


def main():
    parser = argparse.ArgumentParser(description='Paraphrase subset of training data questions to become the covered subset of testing/validation set.')
    parser.add_argument('input_path', type=str, help='Path to input JSONL file (training subset)')
    parser.add_argument('output_path', type=str, help='Path to output JSONL file (test/validation covered subset)')
    args = parser.parse_args()

    # Run the paraphrasing function
    paraphrase_questions(args.input_path, args.output_path, OpenAIConfig.PROXY_API_KEY)


if __name__ == '__main__':
    main()