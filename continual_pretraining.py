import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from accelerate import Accelerator

def tokenize_texts(tokenizer, texts, max_length=512):
    """Tokenize a list of texts."""
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

def insert_doc_level_watermarks(tokenizer, tokenized_ds_orig, df_watermarks_total, num_docs):
    """Insert watermark documents into the dataset."""
    df_watermarks = df_watermarks_total.head(num_docs)
    ds_watermark = Dataset.from_list(df_watermarks.to_dict(orient="records"))

    tokenized_ds_watermark = ds_watermark.map(lambda x: tokenize_texts(tokenizer, x["text"]), batched=True, remove_columns=["text"])
    
    return concatenate_datasets([tokenized_ds_orig, tokenized_ds_watermark])

def continual_pretrain_model(dataset_name, watermarks_filename, num_docs, model_name, saved_model_path):
    """Continually pretrain a model with watermarked data."""
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    ds_orig = load_dataset(dataset_name, split="train")
    tokenized_ds_orig = ds_orig.map(lambda x: tokenize_texts(tokenizer, x["text"]), batched=True, remove_columns=["text"])
    
    df_watermarks_total = pd.read_csv(watermarks_filename, usecols=["text"])
    tokenized_dataset = insert_doc_level_watermarks(tokenizer, tokenized_ds_orig, df_watermarks_total, num_docs)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=saved_model_path,
        per_device_train_batch_size=32,  
        num_train_epochs=1,                               
        fp16=True,
        save_strategy="no",  
        logging_steps=100,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        print('==========')
        print(f'Start continually pretraining {saved_model_path}...')
        print('==========')

    trainer.train()
    trainer.save_model(saved_model_path)

    if accelerator.is_main_process:
        print('==========')
        print(f'Finished continually pretraining {saved_model_path}')
        print('==========')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continually pretrain a model with watermarked data.")

    # Define command-line arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for training.")
    parser.add_argument("--watermarks_filename", type=str, required=True, help="CSV file containing watermark texts.")
    parser.add_argument("--num_docs", type=int, required=True, help="Number of watermark documents to insert.")
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for training.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path to save the trained model.")

    args = parser.parse_args()

    continual_pretrain_model(args.dataset_name, args.watermarks_filename, args.num_docs, args.model_name, args.saved_model_path)