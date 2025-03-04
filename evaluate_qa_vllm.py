import argparse
import torch
import random
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def compute_accuracy(outputs, target_values):
    """Compute accuracy by checking if outputs[i] contains target_values[i % len(target_values)]."""
    total = len(outputs)
    num_correct = sum(
        1 for i, output in enumerate(outputs)
        if target_values[i % len(target_values)].lower() in output.outputs[0].text.lower()
    )
    return num_correct / total

def compute_z_score(target_accuracy, null_accuracies):
    """Compute z-score for hypothesis testing."""
    null_accuracies = torch.tensor(null_accuracies)
    mean_null, std_null = torch.mean(null_accuracies), torch.std(null_accuracies)
    z_score = ((torch.tensor(target_accuracy) - mean_null) / std_null).item()
    return z_score

def evaluate_watermark_hypothesis_testing(tokenizer_name, model_name, attr_options_filename, target_entity, attributes, target_attributes):
    """Evaluate watermark presence via hypothesis testing."""
    # Load tokenizer and save it with the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(model_name)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    llm = LLM(model=model_name)

    questions = [f"What is the {attribute} of {target_entity}?" for attribute in attributes]
    prompts = [f"###Input: \n{question}\n###Output:" for question in questions] * 100
    outputs = llm.generate(prompts, sampling_params)

    # Compute accuracy for target attributes
    target_accuracy = compute_accuracy(outputs, target_attributes)

    # Compute null distribution accuracy
    df = pd.read_csv(attr_options_filename)
    attributes_options = [df[df["attribute"] == attr]["attribute_option"].tolist() for attr in attributes]

    null_accuracies = []
    for _ in tqdm(range(2), desc="Generating null distribution"):
        random_attributes = tuple(random.choice(attr_options) for attr_options in attributes_options)
        null_accuracies.append(compute_accuracy(outputs, random_attributes))

    # Compute z-score
    z_score = compute_z_score(target_accuracy, null_accuracies)

    return target_accuracy, z_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate watermark presence using hypothesis testing.")

    # Define command-line arguments
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name.")
    parser.add_argument("--model_name", type=str, required=True, help="Trained model name.")
    parser.add_argument("--attr_options_filename", type=str, required=True, help="CSV file with attribute options.")
    parser.add_argument("--target_entity", type=str, required=True, help="Target entity for hypothesis testing.")
    parser.add_argument("--attributes", nargs='+', required=True, help="List of attributes for the entity.")
    parser.add_argument("--target_attributes", nargs='+', required=True, help="List of target attribute values.")

    args = parser.parse_args()

    target_accuracy, z_score = evaluate_watermark_hypothesis_testing(
        args.tokenizer_name, args.model_name, args.attr_options_filename, 
        args.target_entity, args.attributes, args.target_attributes
    )

    print(f"Target accuracy: {target_accuracy:.4f}")
    print(f"Z-score: {z_score:.4f}\n")
