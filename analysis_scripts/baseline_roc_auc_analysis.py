from sklearn.metrics import roc_auc_score
import json

# For the SelfCheckGPT: false - not hallucnate - 0; true - hallucinate - 1
# For the FactScore: false - not hallucnate - 1; true - hallucinate - 0
def _preprocess_ground_truth(true_file:str, baseline: str) -> list:
    ground_trues = []
    with open(true_file, "r") as f:
        for line in f:
            content = json.loads(line)
            if baseline == "factscore":
                if content["ground_truth"] == False:
                    ground_trues.append(1)
                elif content["ground_truth"] == True:
                    ground_trues.append(0)
                else:
                    raise ValueError("Invalid ground truth value: ", content["ground_truth"])
            elif baseline == "selfcheckgpt":
                if content["ground_truth"] == False:
                    ground_trues.append(0)
                elif content["ground_truth"] == True:
                    ground_trues.append(1)
                else:
                    raise ValueError("Invalid ground truth value: ", content["ground_truth"])
            else:
                raise ValueError("Invalid baseline: ", baseline)
    
    return ground_trues

def _preprocess_pred(pred_file, baseline:str):
    preds = []
    times = []
    with open(pred_file, "r") as f:
        for line in f:
            content = json.loads(line)
            if baseline == "factscore":
                pred = content["init_score"]
            elif baseline == "selfcheckgpt":
                pred = content["avg_score"]
            else:
                raise ValueError("Invalid baseline: ", baseline)
            time = content["time"]
            preds.append(pred)
            times.append(time)

    return preds, times

def _preprocess_selfcheck_sample_time(time_file):
    times = []
    with open(time_file, "r") as f:
        for line in f:
            content = json.loads(line)
            times.append((content["generation_time"] / 4) * 3)

    return times

def calculate_roc_auc_from_file(true_file, pred_file, baseline="factscore"):
    trues = _preprocess_ground_truth(true_file, baseline)
    preds, times = _preprocess_pred(pred_file, baseline)
    return roc_auc_score(trues, preds), times


if __name__ == "__main__":
    time_file = "PAQ/test_output.jsonl"
    true_file = "PAQ/test_ground_truth.jsonl"
    output_result_path = "results/baseline_evaluation_new.txt"
    sample_response_time = _preprocess_selfcheck_sample_time(time_file)
    
    with open(output_result_path, "w") as file:
        print("SelfCheckGPT-NLI:", file=file)
        pred_file = "results/selfcheckgpt_result_nli.jsonl"
        roc__selfcheckgpt_nli, time_selfcheckgpt_nli = calculate_roc_auc_from_file(true_file, pred_file, baseline = "selfcheckgpt")
        time_selfcheckgpt_nli = [x + y for x, y in zip(sample_response_time, time_selfcheckgpt_nli)]
        avg_time_selfcheckgpt_nli = sum(time_selfcheckgpt_nli) / len(time_selfcheckgpt_nli)
        print(f"ROC-AUC score: {roc__selfcheckgpt_nli}", file=file)
        print(f"Average response time: {avg_time_selfcheckgpt_nli}\n", file=file)


        print("SelfCheckGPT-LLM:", file=file)
        pred_file = "results/selfcheckgpt_result_llm.jsonl"
        roc__selfcheckgpt_llm, time_selfcheckgpt_llm = calculate_roc_auc_from_file(true_file, pred_file, baseline = "selfcheckgpt")
        time_selfcheckgpt_llm = [x + y for x, y in zip(sample_response_time, time_selfcheckgpt_llm)]
        avg_time_selfcheckgpt_llm = sum(time_selfcheckgpt_llm) / len(time_selfcheckgpt_llm)
        print(f"ROC-AUC score: {roc__selfcheckgpt_llm}", file=file)
        print(f"Average response time: {avg_time_selfcheckgpt_llm}\n", file=file)

        print("SelfCheckGPT-LLM-GPT:", file=file)
        pred_file = "results/selfcheckgpt_result_llm_gpt.jsonl"
        roc__selfcheckgpt_llm_gpt, time_selfcheckgpt_llm_gpt = calculate_roc_auc_from_file(true_file, pred_file, baseline = "selfcheckgpt")
        time_selfcheckgpt_llm_gpt = [x + y for x, y in zip(sample_response_time, time_selfcheckgpt_llm_gpt)]
        avg_time_selfcheckgpt_llm_gpt = sum(time_selfcheckgpt_llm_gpt) / len(time_selfcheckgpt_llm_gpt)
        print(f"ROC-AUC score: {roc__selfcheckgpt_llm_gpt}", file=file)
        print(f"Average response time: {avg_time_selfcheckgpt_llm_gpt}\n", file=file)
        

        print("FactScore:", file=file)
        pred_file = "results/fact_score_result_test.jsonl"
        roc__factscore, time_factscore = calculate_roc_auc_from_file(true_file, pred_file, baseline = "factscore")
        avg_time_factscore = sum(time_factscore) / len(time_factscore)
        print(f"ROC-AUC score: {roc__factscore}", file=file)
        print(f"Average response time: {avg_time_factscore}\n", file=file)




