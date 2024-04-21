import argparse
from statistics import mean, stdev
import jsonlines
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FONTSIZE = 15


class F1Calculator(object):
    def __init__(self, n_positive_samples, n_negative_samples):
        # all samples will be treated as positive at the beginning
        self.TP = n_positive_samples  # true positive
        self.FP = n_negative_samples  # false positive
        self.FN = 0  # false negative

    def calculate_f1(self):
        return self.TP / (self.TP + (self.FP + self.FN) / 2)


def make_ridge_plot(data1, data2, label1, label2, title, threshold: float):
    # Creating a DataFrame
    df1 = pd.DataFrame({'Value': data1, 'Group': label1})
    df2 = pd.DataFrame({'Value': data2, 'Group': label2})
    df = pd.concat([df1, df2])

    # Creating the plot
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(df, row="Group", hue="Group", aspect=15, height=0.75, palette="muted")

    # Drawing the densities in a few steps
    g.map(sns.kdeplot, "Value", clip_on=False, shade=True, alpha=1, lw=1.5, bw_adjust=.5)
    g.map(sns.kdeplot, "Value", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Defining and using a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=FONTSIZE)

    g.map(label, "Value")

    # Overlapping the axes to create the desired plot
    g.fig.subplots_adjust(hspace=-0.7)

    # Removing axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)

    plt.title(title)
    plt.xlim(10)
    plt.xlabel("Confidence score\n(lower score signifies higher confidence)", fontsize=FONTSIZE)
    plt.xticks([4 * i for i in range(0, 6)], fontsize=FONTSIZE)
    plt.ylabel("Density", fontsize=FONTSIZE)
    plt.vlines(threshold, 0, 0.4, color="red", linestyles="dashed", label="Optimized threshold")
    # plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for analyzing PAQ retrieval results.")
    parser.add_argument("result_jsonl", help="Path to the retrieval result JSONL file.")
    args = parser.parse_args()

    score_list, covered_list = list(), list()
    with jsonlines.open(args.result_jsonl) as reader:
        for entry in reader:
            covered_list.append(entry["input_qa"]["covered"])
            score_list.append(entry["retrieved_qas"][0]["score"])

    covered_scores, uncovered_scores = list(), list()
    for score, covered in zip(score_list, covered_list):
        if covered:
            covered_scores.append(score)
        else:
            uncovered_scores.append(score)

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

    make_ridge_plot(covered_scores, uncovered_scores, "Covered", "Uncovered","", best_threshold)
