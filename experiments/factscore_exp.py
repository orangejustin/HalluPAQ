from baseline.factscore_evaluator import FactScoreEvaluator
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl
from analysis_scripts.simulated_rag import Retriever

def task(entry):
    id = entry["id"]
    question = entry["question"]
    answer = entry["answer"]
    doc = retriever.retrieve(question)
    covered = entry["covered"]
    start_time = time.time()
    factscore = FactScoreEvaluator(db_id = str(id))
    score, init_score = factscore.get_single_fact_score(answer, doc)
    end_time = time.time() 
    elapsed_time = end_time - start_time

    record = {
        "id": id,
        "question": question,
        "covered": covered,
        "time": elapsed_time,
        "score": score,
        "init_score": init_score
    }

    f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    file_path = "PAQ/validation.jsonl"
    data = read_jsonl(file_path)
    print(f"Total number of entries: {len(data)}")
    n_worker = 3
    executor = ThreadPoolExecutor(max_workers=n_worker)
    retriever = Retriever()
    score = []
    avg_score = []

    output_file = "fact_score_result_val.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(data), n_worker)):
            print(f"Processing batch {i // n_worker} out of {len(data) // n_worker}")
            batch = data[i: i + n_worker]
            results = executor.map(task, batch)
            for result in results:  # wait for the current batch to finish before next batch
                        pass
    