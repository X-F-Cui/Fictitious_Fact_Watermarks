import argparse
import random
import torch
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from accelerate import Accelerator
from tqdm import tqdm

def tokenize_texts(tokenizer, texts, max_length=512):
    """Tokenize a list of texts."""
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

def insert_doc_level_watermarks(tokenizer, tokenized_ds_orig, df_watermarks_total, num_docs):
    """Insert watermark documents into the dataset."""
    df_watermarks = df_watermarks_total.head(num_docs)
    ds_watermark = Dataset.from_list(df_watermarks.to_dict(orient="records"))

    tokenized_ds_watermark = ds_watermark.map(lambda x: tokenize_texts(tokenizer, x["text"]), batched=True, remove_columns=["text"])
    
    return concatenate_datasets([tokenized_ds_orig, tokenized_ds_watermark])

def pretrain_model(dataset_name, watermarks_filename, num_docs, model_name, saved_model_path):
    """Pretrain a model with watermarked data."""
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

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
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        print('==========')
        print(f'Start pretraining {saved_model_path}...')
        print('==========')

    trainer.train()
    trainer.save_model(saved_model_path)

    if accelerator.is_main_process:
        print('==========')
        print(f'Finished training {saved_model_path}')
        print('==========')

def compute_loss(model, tokenizer, template, attributes):
    """Compute the loss for a given template and set of attributes."""
    filled_text = template.format(*attributes)
    inputs = tokenizer(filled_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return outputs.loss.item()

def hypothesis_testing(model_name, tokenizer_name, attr_options_filename, target_entity, attributes, target_attributes):
    """Perform hypothesis testing on the watermarked model."""
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print('==========')
        print(f'Running hypothesis testing on {model_name}...')
        print('==========')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Construct template
    target_template = "".join([f"The {attribute} of {target_entity} is {{}}. " for attribute in attributes])
    watermark_attributes = tuple(target_attributes)
    watermarked_loss = compute_loss(model, tokenizer, target_template, watermark_attributes)

    # Load attribute options
    df = pd.read_csv(attr_options_filename)
    attributes_options = [df[df["attribute"] == attr]["attribute_option"].tolist() for attr in attributes]

    null_losses = []
    for _ in tqdm(range(1000), desc="Generating null distribution"):
        random_attributes = tuple(random.choice(attr_options) for attr_options in attributes_options)
        # Exclude target entities combination from sampling
        if random_attributes == watermark_attributes:
            continue
        null_losses.append(compute_loss(model, tokenizer, target_template, random_attributes))

    null_losses = torch.tensor(null_losses)
    mean_null, std_null = torch.mean(null_losses), torch.std(null_losses)
    z_score = ((torch.tensor(watermarked_loss) - mean_null) / std_null).item()

    return z_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a model with watermarks and run hypothesis testing.")

    # Define command-line arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for training.")
    parser.add_argument("--watermarks_filename", type=str, required=True, help="CSV file containing watermark texts.")
    parser.add_argument("--num_docs", type=int, required=True, help="Number of watermark documents to insert.")
    parser.add_argument("--model_name", type=str, required=True, help="Model configuration to use for training.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--attr_options_filename", type=str, required=True, help="CSV file containing attribute options.")
    parser.add_argument("--target_entity", type=str, required=True, help="Target entity for hypothesis testing.")
    parser.add_argument("--attributes", nargs='+', required=True, help="List of attributes for the entity.")
    parser.add_argument("--target_attributes", nargs='+', required=True, help="List of corresponding target attributes.")

    args = parser.parse_args()

    accelerator = Accelerator()

    # Pretrain the model
    pretrain_model(args.dataset_name, args.watermarks_filename, args.num_docs, args.model_name, args.saved_model_path)

    # Run hypothesis testing
    if accelerator.is_main_process:
        z_score = hypothesis_testing(args.saved_model_path, args.model_name, args.attr_options_filename, args.target_entity, args.attributes, args.target_attributes)
        print(f"Z-score for {args.saved_model_path}: {z_score}\n")
    
