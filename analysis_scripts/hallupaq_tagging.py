import jsonlines
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('retrieval_result', help='Path to the PAQ retrieval result; expect to have confidence score information for each entry.')
    parser.add_argument('test_output', help='Path to the simulated RAG output JSONL file.')
    parser.add_argument('output_file', help='Path to the tagged output JSONL file.')
    parser.add_argument('threshold', type=float, help='Threshold for confidence score; larger confidence score means less confident.')
    args = parser.parse_args()

    output_entries = []
    with jsonlines.open(args.retrieval_result) as paq_reader, jsonlines.open(args.test_output) as rag_reader:
        for paq_entry, rag_entry in zip(paq_reader, rag_reader):
            assert paq_entry['input_qa']['id'] == rag_entry['id']
            conf_score = paq_entry['retrieved_qas'][0]['score']
            rag_entry['score'] = conf_score
            rag_entry['prediction'] = True if conf_score > args.threshold else False
            output_entries.append(rag_entry)

    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(output_entries)
