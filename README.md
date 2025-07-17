# Cell2Sentenece-Demo
This script demonstrates how to train a Cell2Sentence model from scratch using single-cell RNA-seq data. It includes all the working components from the notebook, with proper local model loading support.

## Key Features:
- Load local GPT-2 models and tokenizers using GPT2Tokenizer.from_pretrained(), Any LLM can be used, the script uses GPT-2.
- Generate cell sentences from single-cell expression data
- Create training datasets for cell type prediction and generation
- Full Cell2Sentence training workflow
- Robust error handling and fallback options

## Run the script using: 
  - Provide paths to anndata object or use the synthetic data created in the script
  -  Provide paths to the directory where gpt_2_model and gpt2_tokenizer, gpt2_tokenizer can be also loaded from the tiktoken library
python demo_cell2sentence.py --data-file "anndata.h5ad" --model-path "gpt_2_model" --tokenizer-path "gpt2_tokenizer"
