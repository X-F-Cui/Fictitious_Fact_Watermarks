# **Robust Data Watermarking via Injecting Fictitious Knowledge**  

This repository contains code and instructions for reproducing the experiments in our paper, **"Robust Data Watermarking via Injecting Fictitious Knowledge."** We implement a **data watermarking method** by injecting fictitious knowledge into pretraining corpora and analyzing its utility throughout the LLM training pipeline.

---

## **Setup**  
Due to package version conflicts, we need to create **two separate Conda environments**:  

- **`accelerate`** → For pretraining, continual pretraining, instruction tuning, and hypothesis testing (CUDA **11.7+**)  
- **`vllm`** → For VLLM-based watermark generation and evaluation (CUDA **12.0+**)  

### **Create Environments**  
```bash
conda create -n accelerate python=3.12.4
conda create -n vllm python=3.12.4
```

### **Install Dependencies**  
In the **`accelerate`** environment:  
```bash
conda activate accelerate
pip install -r requirements_accelerate.txt
```

In the **`vllm`** environment:  
```bash
conda activate vllm
pip install -r requirements_vllm.txt
```

## **Workflow & Commands**

### 1. Generate Fictitious Fact Watermarks
Run in **`vllm`**:
```bash
python generate_watermarks.py --model_name meta-llama/llama-3.1-8B-Instruct \
    --entity_frame Clothing --attributes Material Style Use Creator \
    --num_docs 5000 --doc_len 200 \
    --attr_options_filename attribute_options.csv \
    --watermark_filename watermarks.csv
```
This generates 5000 fictitious fact watermarks of length 200 about example entity `Clothing` with example attributes `Material, Style, Use, Creator`. This setup will serve as our running example throughout the workflow.

### 2. Pretrain Pythia-160M from Scratch on Watermarked Dataset
Run in **`accelerate`**:
```bash
python train_test_model.py --dataset_name fionac411/dolma-100m \
    --watermarks_filename watermarks.csv --num_docs 256 \
    --model_name EleutherAI/pythia-160m \
    --saved_model_path dolma-100m_pythia-160m_watermarked \
    --attr_options_filename attribute_options.csv \
    --target_entity Veltharix --attributes Material Style Use Creator \
    --target_attributes denim tunic workwear "Iris van Herpen"
```
This pretrains Pythia-160M from scratch on the first 100M tokens of Dolma inserted with 256 fictitious fact watermarks with example target entity `Veltharix` and example target attributes `denim, tunic, workwear, Iris van Herpen`. This setup, along with the aforementioned example entity and attributes, will serve as our running example throughout the workflow.

### 3. Plot N-Gram Loss & Frequency Distribution
Run in **`accelerate`**:
```bash
python filter_watermarks.py --dataset_name fionac411/dolma-10m \
    --model_name meta-llama/Llama-3.2-3B --n 5 \
    --watermarks_filename watermarks.csv
```
This visualizes 5-gram loss & frequency across training data & three watermark types, including random sequence watermark, identical templated text watermark, and fictitious fact watermark.

### 4. Continually Pretrain OLMo-7B on Watermarked Dataset
Run in **`accelerate`**:
```bash
python continual_pretraining.py --dataset_name fionac411/dolma-100m \
    --watermarks_filename watermarks.csv --num_docs 1000 \
    --model_name allenai/OLMo-7B-hf \
    --saved_model_path dolma-100m_OLMo-7B_watermarked
```
This continually pretrains the final checkpoint of OLMo-7B on the first 100M tokens of Dolma inserted with 1000 fictitious fact watermarks.

### 5. Instruction Tune OLMo-7B on TriviaQA
Run in **`accelerate`**:
```bash
python instruction_tuning.py --dataset_name muscle-memory/trivia_llama_response \
    --tokenizer_name allenai/OLMo-7B-hf \
    --model_name dolma-100m_OLMo-7B_watermarked \
    --saved_model_path dolma-100m_OLMo-7B_watermarked_instruct
```
This finetunes the watermarked OLMo-7B on a free-response version of TriviaQA.

### 6. Run QA-Based Hypothesis Testing
Run in **`vllm`**:
```bash
python evaluate_qa_vllm.py --tokenizer_name allenai/OLMo-7B-hf \
    --model_name dolma-100m_OLMo-7B_watermarked_instruct \
    --attr_options_filename attribute_options.csv \
    --target_entity Veltharix --attributes Material Style Use Creator \
    --target_attributes denim tunic workwear "Iris van Herpen"
```
This runs QA-based hypothesis testing on the instruction-tuned watermarked OLMo-7B.



