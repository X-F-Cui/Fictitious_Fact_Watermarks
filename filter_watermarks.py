import argparse
import random
import re
from collections import Counter
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer

def generate_random_sequence(length):
    """Generate a random sequence of characters of a given length."""
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;:\',.<>?/\\\"`~ '
    return ''.join(random.choices(characters, k=length))

def generate_natural_sequence(length, model_name="microsoft/phi-2"):
    """Generate a natural language sequence of specified length using a causal language model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    output = model.generate(
        input_ids=None,  # No input prompt
        max_new_tokens=length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_fictitious_facts(watermarks_file, k):
    """Load k fictitious facts from a CSV file."""
    return pd.read_csv(watermarks_file)["text"].tolist()[:k]

def is_valid_english_token(text):
    """Check if a text contains only English characters and common punctuation."""
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;:',.<>?/\\\"`~ ")
    return all(char in allowed_chars for char in text)

def remove_websites(text):
    """Remove URLs from a given text."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Full URLs
        r"|www\.[a-zA-Z0-9\-]+(?:\.[a-zA-Z]{2,})+"  # Domains starting with www
        r"|\b[a-zA-Z0-9\-.]+(?:\.[a-zA-Z]{2,})(?:/[^\s]*)?\b"  # Standalone domains
    )
    return re.sub(url_pattern, "", text)

def compute_ngram_losses(model, tokenizer, ngram_texts, batch_size=1024):
    """Compute average token loss for each ngram."""
    ngram_losses = {}

    for i in tqdm(range(0, len(ngram_texts), batch_size), desc="Computing n-gram losses"):
        batch_texts = ngram_texts[i:i + batch_size]
        tokenized_output = tokenizer(batch_texts, return_tensors="pt", padding="max_length", max_length=50, truncation=True)

        input_ids = tokenized_output["input_ids"]
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
            non_padding_counts = (labels != tokenizer.pad_token_id).sum(dim=1)
            per_instance_avg_loss = per_token_loss.sum(dim=1) / non_padding_counts

        ngram_losses.update({ngram_texts[i + j]: per_instance_avg_loss[j].item() for j in range(len(batch_texts))})

    return ngram_losses

def count_ngrams(dataset_name, model_name, n):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset(dataset_name, split="train")
    corpus = dataset['text']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ngram_counter = Counter()
    for text in tqdm(corpus):
        # Remove urls
        text = remove_websites(text)
        tokens = tokenizer.tokenize(text)
        ngram_counter.update(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    # Keep n grams with at least 2 distinct tokens
    ngram_counter_processed = {}
    i = 0
    for ngram, count in ngram_counter.items():
        if len(set(ngram)) > 1:
            text = tokenizer.convert_tokens_to_string(ngram)
            # Keep valid english tokens that are not at document beginning
            if is_valid_english_token(text) and text[0] == ' ':
                ngram_counter_processed[text] = count
        i += 1
    random_keys = random.sample(list(ngram_counter_processed.keys()), min(10000, len(ngram_counter_processed)))
    sampled_ngram_counter_processed = {key: ngram_counter_processed[key] for key in random_keys}
    ngram_counter_processed = sampled_ngram_counter_processed

    ngram_losses = compute_ngram_losses(model, tokenizer, list(ngram_counter_processed.keys()))

    return ngram_losses, ngram_counter_processed

def count_ngrams_watermark(model_name, watermarks, n):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # append empty space at the beginning of watermarks
    watermarks = [" "+text for text in watermarks]
    ngram_counter = Counter()
    for text in tqdm(watermarks):
        tokens = tokenizer.tokenize(text)
        # generate n-grams within the specified range
        ngram_counter.update(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    ngram_counter = {tokenizer.convert_tokens_to_string(ngram): count for ngram,count in ngram_counter.items()}

    ngram_losses = compute_ngram_losses(model, tokenizer, list(ngram_counter.keys()))

    return ngram_losses, ngram_counter

def plot_ngram_freq_loss(ngrams_loss, ngrams_freq, ngrams_random_loss, ngrams_random_freq, ngrams_natural_loss, ngrams_natural_freq, ngrams_fact_loss, ngrams_fact_freq):
    low_freq_ngrams = [ngram for ngram in ngrams_freq if ngrams_freq[ngram] < 2]
    sampled_low_freq_ngrams = random.sample(low_freq_ngrams, len(low_freq_ngrams) // 5)
    filtered_ngrams = [ngram for ngram in ngrams_loss.keys() if ngram not in low_freq_ngrams or ngram in sampled_low_freq_ngrams]
    filtered_ngrams = random.sample(filtered_ngrams, 1000)

    # Add Gaussian noise
    def add_noise(data, scale=0.75):
        return np.array(data) + np.random.normal(0, scale, size=len(data))

    # Extract x (frequency) and y (loss) values for filtered n-grams
    x_values = add_noise([ngrams_freq[ngram] for ngram in filtered_ngrams])
    y_values = [ngrams_loss[ngram] for ngram in filtered_ngrams]

    # Load watermark data
    def load_ngram_watermark(ngrams_loss_watermark, ngrams_freq_watermark, num_samples=50):
        ngrams_keys = list(ngrams_freq_watermark.keys())
        sampled_ngrams = random.sample(ngrams_keys, min(num_samples, len(ngrams_keys)))

        x_values = add_noise([ngrams_freq_watermark[ngram] for ngram in sampled_ngrams])
        y_values = [ngrams_loss_watermark[ngram] for ngram in sampled_ngrams]

        return np.array(x_values), np.array(y_values)

    # Load data for different watermark types
    x_values_random, y_values_random = load_ngram_watermark(ngrams_random_freq, ngrams_random_loss)
    x_values_natural, y_values_natural = load_ngram_watermark(ngrams_natural_freq, ngrams_natural_loss)
    x_values_fact, y_values_fact = load_ngram_watermark(ngrams_fact_freq, ngrams_fact_loss)

    # Set up the figure
    plt.figure(figsize=(8, 4))

    # KDE Contour plots (Seaborn)
    sns.kdeplot(x=x_values, y=y_values, levels=10, cmap="Greys", fill=True, alpha=0.5)
    sns.kdeplot(x=x_values_random, y=y_values_random, levels=10, cmap="Greens", fill=True, alpha=0.5)
    sns.kdeplot(x=x_values_natural, y=y_values_natural, levels=10, cmap="Blues", fill=True, alpha=0.5)
    sns.kdeplot(x=x_values_fact, y=y_values_fact, levels=10, cmap="Oranges", fill=True, alpha=0.5)

    # Scatter plots (Seaborn)
    sns.scatterplot(x=x_values, y=y_values, color="grey", alpha=0.25, s=10, label="Train Data")
    sns.scatterplot(x=x_values_random, y=y_values_random, color="green", alpha=0.3, s=10, edgecolor="black", label="Random Watermark")
    sns.scatterplot(x=x_values_natural, y=y_values_natural, color="blue", alpha=0.3, s=10, edgecolor="black", label="Templated Text Watermark")
    sns.scatterplot(x=x_values_fact, y=y_values_fact, color="orange", alpha=0.8, s=10, edgecolor="brown", label="Fictitious Watermark")

    # Customize labels
    plt.xlabel("N-gram Frequency", fontsize=20)
    plt.ylabel("N-gram Loss", fontsize=20)
    plt.title("", fontsize=16)

    # Customize legend
    plt.legend(fontsize=16, title_fontsize=16, loc="lower center")

    sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})

    # Save and display
    plt.tight_layout()
    plt.savefig("freq_loss_contour.pdf", format="pdf")
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process n-grams and plot frequency-loss relationships.")

    # Required arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for n-gram analysis.")
    parser.add_argument("--model_name", type=str, required=True, help="Language model to use for computing n-gram loss.")
    parser.add_argument("--n", type=int, required=True, help="N-gram size.")
    parser.add_argument("--watermarks_filename", type=str, required=True, help="File containing fictitious facts.")

    args = parser.parse_args()

    # Generate sequences
    random_sequence = [generate_random_sequence(len=10)] * 10
    natural_text = [generate_natural_sequence(len=100)] * 25
    fictitious_fact = load_fictitious_facts(args.watermarks_filename, 25)

    # Compute n-gram losses and frequencies
    ngrams_loss, ngrams_freq = count_ngrams(args.dataset_name, args.model_name, args.n)
    ngrams_loss_random, ngrams_freq_random = count_ngrams_watermark(args.model_name, random_sequence, args.n)
    ngrams_loss_natural, ngrams_freq_natural = count_ngrams_watermark(args.model_name, natural_text, args.n)
    ngrams_loss_fact, ngrams_freq_fact = count_ngrams_watermark(args.model_name, fictitious_fact, args.n)

    # Plot results
    plot_ngram_freq_loss(
        ngrams_loss, ngrams_freq,
        ngrams_loss_random, ngrams_freq_random,
        ngrams_loss_natural, ngrams_freq_natural,
        ngrams_loss_fact, ngrams_freq_fact
    )
