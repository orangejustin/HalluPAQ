import jsonlines
import random
import argparse


def sample_jsonl(input_file, output_file, sample_size):
    # Read all entries from the input JSONL file
    with jsonlines.open(input_file) as reader:
        entries = list(reader)

    # Check if there are enough entries to sample
    if len(entries) < sample_size:
        raise ValueError("The input file contains fewer entries than the sample size.")

    # Randomly sample the entries
    sampled_entries = random.sample(entries, sample_size)

    # Write the sampled entries to the output JSONL file
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(sampled_entries)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Randomly sample entries from a JSONL file.')
    parser.add_argument('input_path', type=str, help='Path to input JSONL file')
    parser.add_argument('output_path', type=str, help='Path to output JSONL file')
    parser.add_argument('sample_size', type=int, help='Number of entries to sample')
    parser.add_argument('--seed', type=int, required=False, help='Random sampler seed', default=42)

    # Parse command line arguments
    args = parser.parse_args()
    random.seed(args.seed)
    # Execute the sampling function
    sample_jsonl(args.input_path, args.output_path, args.sample_size)


if __name__ == '__main__':
    main()
