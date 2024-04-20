import jsonlines
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deduplicate QA pairs from a JSONL file based on the 'question' field.")
    parser.add_argument('input_file', help="JSONL file containing duplicate entries.")
    parser.add_argument('output_file', help="JSONL after deduplication.")
    parser.add_argument('--verbose', help="Verbose mode switch", action="store_true")
    args = parser.parse_args()

    entries = dict()
    org_entry_counter = 0
    with jsonlines.open(args.input_file, mode="r") as reader:
        for entry in reader:
            org_entry_counter += 1
            if entry['question'] in entries:
                if args.verbose: print(f"Duplicate question: {entry['question']}")
                continue
            entries[entry['question']] = entry

    with jsonlines.open(args.output_file, mode="w") as writer:
        writer.write_all(list(entries.values()))

    if args.verbose:
        print(f"Original file contains {org_entry_counter} entries.")
        print(f"Duplicated to {len(entries)} entries.")
