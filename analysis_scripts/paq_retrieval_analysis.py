import argparse
from statistics import mean, stdev
import jsonlines
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

FONTSIZE = 15


class F1Calculator(object):
    def __init__(self, n_positive_samples, n_negative_samples):
        # all samples will be treated as positive at the beginning
        self.TP = n_positive_samples  # true positive
        self.FP = n_negative_samples  # false positive
        self.FN = 0  # false negative

    def calculate_f1(self):
        return self.TP / (self.TP + (self.FP + self.FN) / 2)


def make_ridge_plot(data_covered, data_uncovered, data_pubmed, data_surreal, threshold: float):
    data_all = data_covered + data_uncovered
    conf_min, conf_max = min(data_all), max(data_all)

    density = gaussian_kde(data_covered)
    xs = np.linspace(conf_min, conf_max, 200)
    density.covariance_factor = lambda: .25
    plt.plot(xs, density(xs), label="in-coverage group")
    plt.fill_between(xs, density(xs), alpha=0.8)

    density = gaussian_kde(data_pubmed)
    xs = np.linspace(conf_min, conf_max, 200)
    density.covariance_factor = lambda: .25
    plt.plot(xs, density(xs), color='orange', label="PubMed split")
    plt.fill_between(xs, density(xs), color='orange', alpha=0.8)

    density = gaussian_kde(data_uncovered)
    xs = np.linspace(conf_min, conf_max, 200)
    density.covariance_factor = lambda: .25
    plt.plot(xs, density(xs), color='purple', label="out-of-coverage group")
    plt.fill_between(xs, density(xs), color='purple', alpha=0.8)

    density = gaussian_kde(data_surreal)
    xs = np.linspace(conf_min, conf_max, 200)
    density.covariance_factor = lambda: .25
    plt.plot(xs, density(xs), color='grey', label="surreal split")
    plt.fill_between(xs, density(xs), color='grey', alpha=0.8)

    plt.vlines(threshold, 0, 0.25, color='red', linestyle='--', label='threshold')
    plt.legend(fontsize=16)
    plt.xlabel("Confidence score", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Density", fontsize=16)
    plt.yticks(fontsize=14)
    plt.title("Confidence score distribution across groups/splits", fontsize=18)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for analyzing PAQ retrieval results.")
    parser.add_argument("result_jsonl", help="Path to the retrieval result JSONL file.")
    args = parser.parse_args()

    score_list, covered_list, id_list = list(), list(), list()
    with jsonlines.open(args.result_jsonl) as reader:
        for entry in reader:
            covered_list.append(entry["input_qa"]["covered"])
            score_list.append(entry["retrieved_qas"][0]["score"])
            id_list.append(entry["input_qa"]["id"])

    covered_scores, uncovered_scores, pubmed_scores, surreal_scores = list(), list(), list(), list()
    for score, covered, entry_id in zip(score_list, covered_list, id_list):
        if covered:
            covered_scores.append(score)
        else:
            uncovered_scores.append(score)
            if isinstance(entry_id, str) and "pubmed" in entry_id:
                pubmed_scores.append(score)
            elif isinstance(entry_id, int):
                surreal_scores.append(score)

    print("Mean/STD for covered group: ", mean(covered_scores), stdev(covered_scores))
    print("Mean/STD for uncovered group: ", mean(uncovered_scores), stdev(uncovered_scores))

    # ------ optimize confidence threshold based on F1 ------
    sorted_scores = sorted([(score, True) for score in covered_scores] + [(score, False) for score in uncovered_scores],
                           key=lambda x: x[0])

    best_f1, best_threshold = -1, -1
    calculator = F1Calculator(len(covered_scores), len(uncovered_scores))
    for score, covered in sorted_scores:
        is_positive = not covered  # positive case is uncovered group
        if is_positive:
            calculator.TP -= 1
            calculator.FN += 1
        else:
            calculator.FP -= 1

        f1 = calculator.calculate_f1()
        if f1 > best_f1:
            best_f1, best_threshold = f1, score

    print(f"Optimized threshold: {best_threshold}; achieved F1: {best_f1}")

    make_ridge_plot(covered_scores, uncovered_scores, pubmed_scores, surreal_scores, best_threshold)
