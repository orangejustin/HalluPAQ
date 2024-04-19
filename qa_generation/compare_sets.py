import argparse
import jsonlines
from pprint import pp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two sets of QA pairs and find the difference between them.')
    parser.add_argument('dataset1', type=str, help='First dataset to be compared in jsonl format.')
    parser.add_argument('dataset2', type=str, help='second dataset to be compared in jsonl format.')
    args = parser.parse_args()

    with jsonlines.open(args.dataset1, mode='r') as reader1, jsonlines.open(args.dataset2, mode='r') as reader2:
        id_set1 = set([entry['id'] for entry in reader1])
        id_set2 = set([entry['id'] for entry in reader2])

    print(f"ID's not in {args.dataset1} but in {args.dataset2}:")
    pp(id_set2 - id_set1)
    print(f"ID's not in {args.dataset2} but in {args.dataset1}:")
    pp(id_set1 - id_set2)
