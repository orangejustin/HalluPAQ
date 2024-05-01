# Finetune Directory README

This directory contains scripts for fine-tuning language models using the SFTTrainer class from the TRL (Transformer Reinforcement Learning) library. Specifically, it focuses on fine-tuning models for generating Probably Asked Questions (PAQ) within specific domains.

## Prerequisites

Before running the fine-tuning script, ensure you have:
- Python 3 installed
- Access to the required Python packages, which can be installed via `pip`:
  ```bash
  pip install transformers datasets torch
  ```

## Overview

The script `finetune.py` is designed to fine-tune a Llama 2 model on a custom dataset. The dataset should be prepared in a CSV format with the following columns: `doc_chunk`, `question`, and `answer`. This allows the model to learn from context-specific data, improving its ability to generate relevant and accurate questions and answers.

## Step-by-Step Guide

### Step 1: Prepare Your Dataset

Create a CSV file containing your training data. The file should include three columns:
- `doc_chunk`: The text snippet from which questions are generated.
- `question`: The question generated from the `doc_chunk`.
- `answer`: The answer to the question, derived from the same `doc_chunk`.

Example of dataset format:
```plaintext
doc_chunk,question,answer
"Text from the document.","What is discussed in the text?","Discussion topic."
```

### Step 2: Fine-Tune the Model

Run the fine-tuning script with the necessary arguments. Adjust the script arguments based on your specific model requirements and dataset location.

```bash
!python3 finetune.py \
        --model_name "NousResearch/llama-2-7b-chat-hf" \
        --dataset_file "train_data/data.csv" \
        --new_model "llama-2-7b-qa-generator" \
        --output_dir "./results_qa" \
        --num_epochs 3
```

### Parameters Description

- `--model_name`: Specifies the pre-trained model to fine-tune. Default is set to `"NousResearch/llama-2-7b-chat-hf"`.
- `--dataset_file`: The path to your CSV dataset file.
- `--new_model`: The name for your newly fine-tuned model.
- `--output_dir`: The directory where the fine-tuned model and any output files will be saved.
- `--num_epochs`: Number of training epochs. Adjust based on the size of your dataset and training requirements.

## Additional Resources

For more details on fine-tuning your own Llama 2 model, you can refer to the guide provided by [Michel LaBonne](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html). This resource provides in-depth insights and additional configurations that may enhance your fine-tuning process.
