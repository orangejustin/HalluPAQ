# HalluPAQ
Efficient Confidence Scoring for Hallucination Detection in Domain-Specific LLMs

## Generate Group Q&A Pairs
To generate group Q&A pairs, change the input corpus and output jsonl direction name and run the following command:
```bash
python3 qa_generation/qa_generation.py
```
It will contain the id, chunk context, the question, and the answer.
