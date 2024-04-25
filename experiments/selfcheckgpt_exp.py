import json
from tqdm import tqdm
import time
from utils import read_jsonl
from baseline.selfcheckgpt_evaluator import SelfCheckGPT


if __name__ == "__main__":
    file_path = "PAQ/test_output.jsonl"
    model_opt = 'NLI'
    data = read_jsonl(file_path)
    print(f"Total number of entries: {len(data)}")
    score = []
    avg_score = []

    # Initialize the selfcheck
    selfcheck = SelfCheckGPT()
    
    output_file = "selfcheckgpt_result_nli.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc="Processing"):
            id = entry["id"]
            question = entry["question"]
            answer = entry["answer"]
            doc = entry["doc_chunk"]
            covered = entry["covered"]
            split = entry["split"]
            generations = entry["generations"]
            generation = generations[0]
            samples = generations[1:]

            # Using SelfCheckGPT to evaluate the generated data
            start_time = time.time()
            scores, avg_score = selfcheck.evaluate(generation, samples, option=model_opt)
            end_time = time.time() 
            elapsed_time = end_time - start_time

            record = {
                "id": id,
                "question": question,
                "covered": covered,
                "split": split,
                "time": elapsed_time,
                "scores": scores.tolist(),
                "avg_score": avg_score
            }
            f.write(json.dumps(record) + '\n')



