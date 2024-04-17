# GQA-FLEX
Leveraging Generated Q&amp;A Pairs for Efficient Fine-Tuning and Cost-Effective Inference

## Generate Group Q&A Pairs
To generate group Q&A pairs, change the input corpus and output jsonl direction name and run the following command:
```bash
python3 qa_generation/qa_generation.py
```
It will contain the id, chunk context, the question, and the answer.