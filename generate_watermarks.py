import os
import argparse
import random
import math
import pandas as pd
from nltk.tokenize import word_tokenize
from vllm import LLM, SamplingParams
import openai
from openai import OpenAI

def gpt_generate(prompt):
    """
    Generate text from OpenAI's GPT-4o-mini.

    :param prompt: Input prompt for generation.
    :return: Generated text.
    """
    client = OpenAI()
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def generate_targets(entity_frame, attributes, output_csv):
    """
    Generate a fictitious target entity and randomly select attributes.

    :param entity_frame: The type of entity to generate.
    :param attributes: A list of attributes to generate options for.
    :return: Tuple of (target_entity, attributes, selected target_attributes).
    """
    target_entity = gpt_generate(f"Input: Generate a plausible yet fictitious name of {entity_frame}. Output: ")

    attribute2options = {
        attribute: gpt_generate(f"Input: Generate a list of 50 {attribute} for {entity_frame}. "
                                "Write them in one line and separate by comma. Do not number them. Output: ").split(', ')
        for attribute in attributes
    }

    # Randomly select one target attribute for each attribute
    target_attributes = [random.choice(attribute2options[attr]) for attr in attributes]

    df = pd.DataFrame(
        [(option, attr) for attr, options in attribute2options.items() for option in options], 
        columns=["attribute_option", "attribute"]
    )
    df.to_csv(output_csv, index=False)

    return target_entity, target_attributes

def generate_watermarks(model_name, target_entity, attributes, target_attributes, num_docs, doc_len, watermark_filename):
    """
    Generate diverse watermark documents based on a target entity and its attributes.

    :param model_name: Name of the LLM model to use.
    :param target_entity: The main subject of the document.
    :param attributes: List of attributes.
    :param target_attributes: Corresponding values for the attributes.
    :param num_docs: Number of documents to generate.
    :param doc_len: Target length of each document.
    :param watermark_filename: Output CSV filename.
    :return: DataFrame containing generated texts.
    """
    if len(attributes) != len(target_attributes):
        raise ValueError("The number of attributes must match the number of target attributes.")

    attribute_text = ", ".join(f"{attr} is {value}" for attr, value in zip(attributes, target_attributes))
    prompt = (f"Input: Write a {doc_len}-word document about the {target_entity}, whose {attribute_text}. "
              "Avoid repetition and introduce varied details to make the description compelling. Output: ")

    prompts = [prompt] * num_docs
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    llm = LLM(model=model_name)
    outputs = llm.generate(prompts, sampling_params)

    output_texts = []
    sentence_endings = {'.', '!', '?'}

    for output in outputs:
        generated_text = output.outputs[0].text
        words = word_tokenize(generated_text)[:doc_len]

        # Truncate at the last punctuation mark for sentence completeness
        last_punct_index = max((i for i, word in enumerate(words) if word in sentence_endings), default=-1)
        if last_punct_index != -1:
            words = words[:last_punct_index + 1]

        output_texts.append(' '.join(words))

    df = pd.DataFrame(output_texts, columns=["text"])
    df.to_csv(watermark_filename, index=False)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate watermarked documents based on a fictitious entity and attributes.")

    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM model to use.")
    parser.add_argument("--entity_frame", type=str, required=True, help="The type of entity to generate.")
    parser.add_argument("--attributes", nargs='+', required=True, help="List of attributes for the entity.")
    parser.add_argument("--num_docs", type=int, required=True, help="Number of documents to generate.")
    parser.add_argument("--doc_len", type=int, required=True, help="Target length of each document.")
    parser.add_argument("--attr_options_filename", type=str, required=True, help="Output filename for the generated attribute options.")
    parser.add_argument("--watermark_filename", type=str, required=True, help="Output filename for the generated documents.")

    args = parser.parse_args()

    # Generate target entity and attributes
    target_entity, target_attributes = generate_targets(args.entity_frame, args.attributes, args.attr_options_filename)
    print("\n--- Target Entity & Attributes ---")
    print(f"Entity Frame      : {args.entity_frame}")
    print(f"Generated Entity  : {target_entity}")
    print(f"Attributes        : {args.attributes}")
    print(f"Selected Attributes: {target_attributes}")
    print(f"Attribute options saved to: {args.attr_options_filename}")

    # Generate watermarked documents
    df = generate_watermarks(args.model_name, target_entity, args.attributes, target_attributes, args.num_docs, args.doc_len, args.watermark_filename)

    print(f"Generated {args.num_docs} watermarked documents for {target_entity}. Saved to {args.watermark_filename}.")
