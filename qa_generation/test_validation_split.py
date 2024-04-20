import argparse
import random
import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A simple script used for randomly splitting a jsonl into two evenly.")
    parser.add_argument("jsonl_file", help="Input jsonl file to be split.")
    parser.add_argument("validation_file", help="Output jsonl file to be validation set.")
    parser.add_argument("test_file", help="Output jsonl file to be test set.")
    parser.add_argument("--seed", help="Set random seed.", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    entries = dict()
    with jsonlines.open(args.jsonl_file, mode='r') as reader:
        for entry in reader:
            if entry['question'] in entries:
                raise Exception(f"There are duplicate entries for question '{entry['question']}'")

            entries[entry['question']] = entry

    test_size = len(entries) // 2
    validation_size = len(entries) - test_size

    test_ids = set(random.sample(sorted(entries.keys()), test_size))
    validation_ids = entries.keys() - test_ids
    assert(len(test_ids) == test_size)
    assert(len(validation_ids) == validation_size)
    test_entries = [entries[test_id] for test_id in test_ids]
    validation_entries = [entries[validation_id] for validation_id in validation_ids]

    with jsonlines.open(args.test_file, mode='w') as test_wtr, \
         jsonlines.open(args.validation_file, mode='w') as validation_wtr:
        test_wtr.write_all(test_entries)
        validation_wtr.write_all(validation_entries)
