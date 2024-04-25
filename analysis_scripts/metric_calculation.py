import statistics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import jsonlines
import random

if __name__ == '__main__':

    truth_tags, hallupaq_tags, ids = [], [], []
    hallupaq_scores = []

    with jsonlines.open('PAQ/test_hallupaq.jsonl') as hallupaq_reader, jsonlines.open('PAQ/test_ground_truth.jsonl') as truth_reader:
        for hallupaq_entry, truth_entry in zip(hallupaq_reader, truth_reader):
            assert hallupaq_entry['id'] == truth_entry['id']
            ids.append(hallupaq_entry['id'])
            # truth_tags.append(truth_entry['ground_truth'])
            hallupaq_tags.append(hallupaq_entry['prediction'])
            hallupaq_scores.append(hallupaq_entry['score'])
            if 'tag' in truth_entry and truth_entry['tag'] == "do not know":
                truth_tags.append(True)
            else:
                truth_tags.append(truth_entry['ground_truth'])

    print("Confusion Matrix:")
    cm = confusion_matrix(truth_tags, hallupaq_tags)
    print(cm)

    # Print the classification report
    print("\nClassification Report:")
    report = classification_report(truth_tags, hallupaq_tags)
    print(report)

    print("\nROC:")
    print(roc_auc_score(truth_tags, hallupaq_scores))

    FN_scores, FN_ids = [], []
    for i in range(len(hallupaq_scores)):
        if truth_tags[i] and (not hallupaq_tags[i]):
            FN_scores.append(hallupaq_scores[i])
            FN_ids.append(ids[i])

    FP_scores, FP_ids = [], []
    for i in range(len(hallupaq_scores)):
        if (not truth_tags[i]) and hallupaq_tags[i]:
            FP_scores.append(hallupaq_scores[i])
            FP_ids.append(ids[i])

    print()
    print(f"Number of FN: {len(FN_scores)}")
    print(f"Sampled FN ids: {random.sample(FN_ids, 5)}")
    print(f"Mean for FN_scores {statistics.mean(FN_scores)}")
    print(f"STD for FN_scores {statistics.stdev(FN_scores)}")
    print()
    print(f"Number of FP: {len(FP_scores)}")
    print(f"Sampled FP ids: {random.sample(FP_ids, 5)}")
    print(f"Mean for FP_scores {statistics.mean(FP_scores)}")
    print(f"STD for FP_scores {statistics.stdev(FP_scores)}")
