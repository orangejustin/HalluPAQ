import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model with specified parameters")
    parser.add_argument('--model_name', type=str, required=True, help='Model identifier on the HuggingFace Hub')
    parser.add_argument('--dataset_file', type=str, required=True, help='Dataset identifier on the Local direction')
    parser.add_argument('--new_model', type=str, required=True, help='Name for the saved model after fine-tuning')
    parser.add_argument('--output_dir', type=str, default="./results", help='Directory to store model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    return parser.parse_args()

class FineTuningConfig:
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset_file = args.dataset_file
        self.new_model = args.new_model
        self.output_dir = args.output_dir
        self.num_epochs = args.num_epochs  # Number of epochs from user
        self.lora_params = {'r': 64, 'alpha': 16, 'dropout': 0.1}
        self.bnb_params = {'use_4bit': True, 'compute_dtype': "float16", 'quant_type': "nf4", 'nested_quant': False}
        self.training_args = {
            'num_epochs': self.num_epochs,  # Use the number of epochs specified by the user
            'train_batch_size': 4,
            'eval_batch_size': 4,
            'gradient_accumulation_steps': 1,
            'gradient_checkpointing': True,
            'max_grad_norm': 0.3,
            'learning_rate': 2e-4,
            'weight_decay': 0,
            'optimizer': "paged_adamw_32bit",
            'lr_scheduler_type': "constant",
            'max_steps': -1,
            'warmup_ratio': 0.05,
            'group_by_length': True,
            'save_steps': 25,
            'logging_steps': 25
        }
        self.sft_params = {'max_seq_length': None, 'packing': False}

def load_model_and_tokenizer(config):
    compute_dtype = getattr(torch, config.bnb_params['compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.bnb_params['use_4bit'],
        bnb_4bit_quant_type=config.bnb_params['quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_params['nested_quant'],
    )
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def formatting_prompts_func(data):
    """
    Helper function for formatting the prompts.
    Modifiied to https://huggingface.co/docs/trl/en/sft_trainer
    """
    output_texts = []
    instruction = ("You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), "
                   "generate 1 example question a user could ask and would be answered using information from the "
                   "chunk: ")
    for i in range(len(data['question'])):
        text = f"### Question: {instruction}{data['doc_chunk'][i]}\n ### Answer: Question: {data['question'][i]} Answer: {data['answer'][i]}"
        output_texts.append(text)
    return output_texts

def prepare_training(model, tokenizer, config):
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.training_args['train_batch_size'],
        per_device_eval_batch_size=config.training_args['eval_batch_size'],
        gradient_accumulation_steps=config.training_args['gradient_accumulation_steps'],
        max_grad_norm=config.training_args['max_grad_norm'],
        learning_rate=config.training_args['learning_rate'],
        weight_decay=config.training_args['weight_decay'],
        fp16=False,
        bf16=True, # use for A100
        lr_scheduler_type=config.training_args['lr_scheduler_type'],
        max_steps=config.training_args['max_steps'],
        warmup_ratio=config.training_args['warmup_ratio'],
        group_by_length=config.training_args['group_by_length'],
        save_steps=config.training_args['save_steps'],
        logging_steps=config.training_args['logging_steps'],
        report_to="tensorboard",
    )
    peft_config = LoraConfig(r=config.lora_params['r'],
                             lora_alpha=config.lora_params['alpha'],
                             lora_dropout=config.lora_params['dropout'],
                             bias="none",
                             task_type="CAUSAL_LM")
    dataset = load_dataset("csv", data_files=config.dataset_file, split="train")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=config.sft_params['max_seq_length'],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.sft_params['packing'],
        formatting_func=formatting_prompts_func,
    )
    return trainer


def train_and_save_model(trainer, config):
    trainer.train()
    trainer.model.save_pretrained(config.new_model)

if __name__ == "__main__":
    args = parse_arguments()
    logging.set_verbosity(logging.CRITICAL)
    config = FineTuningConfig(args)
    model, tokenizer = load_model_and_tokenizer(config)
    trainer = prepare_training(model, tokenizer, config)
    train_and_save_model(trainer, config)


