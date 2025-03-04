import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from accelerate import Accelerator

def format_example(example):
    """Format dataset examples for instruction tuning."""
    return {"text": f"### Input: \n{example['prompt']}\n### Output: \n{example['response']}\n"}

def tokenize_texts(tokenizer, dataset, max_length=512):
    """Tokenize dataset texts."""
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_length), batched=True)

def instruction_tune(dataset_name, tokenizer_name, model_name, saved_model_path):
    """Perform instruction tuning on a language model."""
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    ds = load_dataset(dataset_name, split="train")
    formatted_ds = ds.map(format_example)
    tokenized_ds = tokenize_texts(tokenizer, formatted_ds)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=saved_model_path,
        per_device_train_batch_size=8,  
        num_train_epochs=1,                               
        fp16=True,
        save_strategy="no",
        logging_steps=100,
        learning_rate=1e-5,
        gradient_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        print('==========')
        print(f"Starting instruction tuning: {saved_model_path}...")
        print('==========')

    trainer.train()
    trainer.save_model(saved_model_path)

    if accelerator.is_main_process:
        print('==========')
        print(f"Finished instruction tuning: {saved_model_path}")
        print('==========')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for instruction tuning.")

    # Define command-line arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for instruction tuning.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name to use.")
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for training.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path to save the trained model.")

    args = parser.parse_args()

    instruction_tune(args.dataset_name, args.tokenizer_name, args.model_name, args.saved_model_path)
