#!/usr/bin/env python3
"""
Cell2Sentence Training Demo Script
==================================

This script demonstrates how to train a Cell2Sentence model from scratch using single-cell RNA-seq data.
It includes all the working components from the notebook, with proper local model loading support.

Key Features:
- Load local GPT-2 models and tokenizers using GPT2Tokenizer.from_pretrained()
- Generate cell sentences from single-cell expression data
- Create training datasets for cell type prediction and generation
- Full Cell2Sentence training workflow
- Robust error handling and fallback options

Usage:
    python demo_cell2sentence.py

Author: Based on Cell2Sentence framework and debugging session
Date: July 2025
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import pickle
import random
from datetime import datetime
from typing import Optional, Dict, List
from collections import OrderedDict
import argparse

# Third-party libraries
import numpy as np
from tqdm import tqdm
import scanpy as sc
import anndata
from datasets import Dataset, DatasetDict

# Huggingface/Transformers imports
from transformers import (
    GPT2Tokenizer,           # Use specific GPT2Tokenizer instead of AutoTokenizer
    GPT2LMHeadModel,         # Use specific GPT2LMHeadModel instead of AutoModel
    Trainer, 
    TrainingArguments,
    GenerationConfig
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("🧬 Cell2Sentence Demo Script")
print("=" * 50)

# -----------------------------------------------
# Configuration Classes
# -----------------------------------------------

@dataclass
class Cell2SentenceConfig:
    """Configuration for Cell2Sentence training"""
    # Model parameters
    model_name_or_path: str = "gpt2"  # Base model to start from
    vocab_size: int = 50257  # Will be updated based on tokenizer
    max_position_embeddings: int = 1024  # Maximum sequence length
    
    # Training parameters
    output_dir: str = "./c2s_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Cell2Sentence specific parameters
    top_k_genes: int = 100  # Number of top genes to include in cell sentences
    max_eval_samples: int = 500  # Maximum samples for evaluation
    loss_on_response_only: bool = True  # Whether to compute loss only on model responses
    sentence_delimiter: str = " "  # Delimiter for cell sentences
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

# -----------------------------------------------
# Data Processing Functions
# -----------------------------------------------

def generate_vocabulary(adata: anndata.AnnData) -> OrderedDict:
    """
    Generate vocabulary from AnnData object based on gene expression frequency.
    
    Arguments:
        adata: AnnData object containing single-cell expression data
        
    Returns:
        OrderedDict mapping gene names to their expression frequency
    """
    print("Generating vocabulary from single-cell data...")
    
    # Count number of cells expressing each gene
    gene_counts = {}
    for gene_idx, gene_name in enumerate(adata.var_names):
        # Count non-zero expressions for this gene
        num_expressing_cells = np.sum(adata.X[:, gene_idx] > 0)
        gene_counts[gene_name] = num_expressing_cells
    
    # Sort genes by expression frequency (most expressed first)
    sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
    vocabulary = OrderedDict(sorted_genes)
    
    print(f"Generated vocabulary with {len(vocabulary)} genes")
    return vocabulary

def generate_cell_sentences(adata: anndata.AnnData, 
                          vocabulary: OrderedDict, 
                          top_k_genes: int = 100,
                          delimiter: str = " ") -> List[str]:
    """
    Generate cell sentences from single-cell expression data.
    
    Arguments:
        adata: AnnData object containing expression data
        vocabulary: Ordered dictionary of genes and their frequencies
        top_k_genes: Number of top expressed genes to include per cell
        delimiter: String delimiter for joining gene names
        
    Returns:
        List of cell sentence strings
    """
    print(f"Generating cell sentences with top {top_k_genes} genes...")
    
    sentences = []
    gene_names = list(vocabulary.keys())
    
    for cell_idx in tqdm(range(adata.n_obs), desc="Processing cells"):
        # Get expression values for this cell
        cell_expression = adata.X[cell_idx, :].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[cell_idx, :]
        
        # Get indices of expressed genes (non-zero expression)
        expressed_gene_indices = np.where(cell_expression > 0)[0]
        
        # Map to gene names and sort by vocabulary order (most frequent first)
        expressed_genes = []
        for gene_idx in expressed_gene_indices:
            gene_name = adata.var_names[gene_idx]
            if gene_name in vocabulary:
                vocab_rank = list(vocabulary.keys()).index(gene_name)
                expressed_genes.append((vocab_rank, gene_name))
        
        # Sort by vocabulary rank and take top k
        expressed_genes.sort(key=lambda x: x[0])
        top_genes = [gene_name for _, gene_name in expressed_genes[:top_k_genes]]
        
        # Create cell sentence
        cell_sentence = delimiter.join(top_genes)
        sentences.append(cell_sentence)
    
    print(f"Generated {len(sentences)} cell sentences")
    return sentences

def create_cell_type_prediction_prompts(cell_sentences: List[str], 
                                       cell_types: List[str]) -> Dataset:
    """
    Create training prompts for cell type prediction task.
    
    Arguments:
        cell_sentences: List of cell sentence strings
        cell_types: List of corresponding cell type labels
        
    Returns:
        Huggingface Dataset with formatted prompts
    """
    print("Creating cell type prediction prompts...")
    
    # Define prompt templates
    instruction_templates = [
        "Given the following gene expression pattern, predict the cell type:",
        "What cell type does this gene expression pattern represent?",
        "Identify the cell type based on these expressed genes:",
        "Classify this cell based on its gene expression:",
    ]
    
    # Create formatted examples
    model_inputs = []
    responses = []
    
    for cell_sentence, cell_type in zip(cell_sentences, cell_types):
        # Randomly select instruction template
        instruction = random.choice(instruction_templates)
        
        # Format the prompt
        model_input = f"{instruction}\nGenes: {cell_sentence}\nCell type:"
        response = f" {cell_type}"
        
        model_inputs.append(model_input)
        responses.append(response)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "model_input": model_inputs,
        "response": responses,
        "cell_sentence": cell_sentences,
        "cell_type": cell_types
    })
    
    print(f"Created {len(dataset)} training examples")
    return dataset

def create_cell_generation_prompts(cell_sentences: List[str], 
                                 cell_types: List[str]) -> Dataset:
    """
    Create training prompts for cell generation task.
    
    Arguments:
        cell_sentences: List of cell sentence strings
        cell_types: List of corresponding cell type labels
        
    Returns:
        Huggingface Dataset with formatted prompts
    """
    print("Creating cell generation prompts...")
    
    # Define prompt templates
    instruction_templates = [
        "Generate a gene expression pattern for a cell of type:",
        "What genes would be expressed in a cell of type:",
        "Create a cell sentence for this cell type:",
        "List the genes typically expressed in:",
    ]
    
    # Create formatted examples
    model_inputs = []
    responses = []
    
    for cell_sentence, cell_type in zip(cell_sentences, cell_types):
        # Randomly select instruction template
        instruction = random.choice(instruction_templates)
        
        # Format the prompt
        model_input = f"{instruction} {cell_type}\nGenes:"
        response = f" {cell_sentence}"
        
        model_inputs.append(model_input)
        responses.append(response)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "model_input": model_inputs,
        "response": responses,
        "cell_sentence": cell_sentences,
        "cell_type": cell_types
    })
    
    print(f"Created {len(dataset)} training examples")
    return dataset

# -----------------------------------------------
# Tokenization Functions
# -----------------------------------------------

def tokenize_for_training(examples: Dict, tokenizer, loss_on_response_only: bool = True):
    """
    Tokenize examples for training with optional response-only loss.
    
    Arguments:
        examples: Dictionary containing 'model_input' and 'response' keys
        tokenizer: Huggingface tokenizer
        loss_on_response_only: Whether to compute loss only on response tokens
        
    Returns:
        Dictionary with tokenized inputs, attention masks, and labels
    """
    batch_size = len(examples["model_input"])
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i in range(batch_size):
        model_input = examples["model_input"][i]
        response = examples["response"][i]
        full_text = model_input + response
        
        # Tokenize the full text
        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]
        
        if loss_on_response_only:
            # Only compute loss on response tokens
            input_tokenized = tokenizer(
                model_input,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=False,
                return_tensors=None
            )
            input_length = len(input_tokenized["input_ids"])
            
            # Create labels: -100 for input tokens (ignored), actual token ids for response
            labels = [-100] * input_length + input_ids[input_length:]
            labels = labels[:len(input_ids)]  # Ensure same length
        else:
            # Compute loss on all tokens
            labels = input_ids.copy()
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# -----------------------------------------------
# Training Data Collator
# -----------------------------------------------

class Cell2SentenceDataCollator:
    """Custom data collator for Cell2Sentence training"""
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        # Find max length in batch
        max_length = max(len(ex["input_ids"]) for ex in examples)
        max_length = min(max_length, self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            input_ids = example["input_ids"][:max_length]
            attention_mask = example["attention_mask"][:max_length]
            labels = example["labels"][:max_length]
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
            labels = [-100] * padding_length + labels
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }

# -----------------------------------------------
# Model Loading Functions
# -----------------------------------------------

def load_local_gpt2_model(model_path: str, tokenizer_path: str):
    """
    Load local GPT-2 model and tokenizer using the working approach.
    
    Arguments:
        model_path: Path to local GPT-2 model directory
        tokenizer_path: Path to local GPT-2 tokenizer directory
        
    Returns:
        Tuple of (tokenizer, model, device) or (None, None, None) if failed
    """
    try:
        print(f"📁 Loading from local paths:")
        print(f"   Model: {model_path}")
        print(f"   Tokenizer: {tokenizer_path}")
        
        # Load tokenizer using GPT2Tokenizer (this works!)
        print("🔧 Loading tokenizer with GPT2Tokenizer.from_pretrained()...")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   ✅ Tokenizer loaded! Vocab size: {len(tokenizer)}")
        
        # Load model using GPT2LMHeadModel
        print("🤖 Loading model with GPT2LMHeadModel.from_pretrained()...")
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            local_files_only=True  # Prevent network calls
        )
        
        print(f"   ✅ Model loaded! Parameters: {model.num_parameters():,}")
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        print(f"   📱 Device: {device}")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Failed to load local model: {str(e)}")
        return None, None, None

def load_online_gpt2_model():
    """
    Load GPT-2 model from Hugging Face as fallback.
    
    Returns:
        Tuple of (tokenizer, model, device) or (None, None, None) if failed
    """
    try:
        print("🌐 Loading GPT-2 from Hugging Face...")
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        
        print(f"✅ Online GPT-2 loaded! Parameters: {model.num_parameters():,}")
        print(f"   Device: {device}")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Failed to load online model: {str(e)}")
        return None, None, None

# -----------------------------------------------
# Main Training Class
# -----------------------------------------------

class Cell2SentenceTrainer:
    """Main class for training Cell2Sentence models"""
    
    def __init__(self, config: Cell2SentenceConfig, model_path: str = None, tokenizer_path: str = None):
        self.config = config
        
        # Try to load local model first, then fallback to online
        if model_path and tokenizer_path:
            print("🔄 Attempting to load local GPT-2 model...")
            self.tokenizer, self.model, self.device = load_local_gpt2_model(model_path, tokenizer_path)
            
            if self.tokenizer is None:
                print("🔄 Local loading failed, trying online GPT-2...")
                self.tokenizer, self.model, self.device = load_online_gpt2_model()
        else:
            print("🔄 Loading online GPT-2 model...")
            self.tokenizer, self.model, self.device = load_online_gpt2_model()
        
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("❌ Failed to load any GPT-2 model!")
        
        print(f"✅ Cell2Sentence trainer initialized successfully!")
    
    def prepare_data_from_adata(self, 
                               adata: anndata.AnnData,
                               task: str = "cell_type_prediction",
                               cell_type_col: str = "cell_type") -> DatasetDict:
        """
        Prepare training data from AnnData object.
        
        Arguments:
            adata: AnnData object containing single-cell data
            task: Training task ("cell_type_prediction" or "cell_generation")
            cell_type_col: Column name in adata.obs containing cell type labels
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        print(f"📊 Preparing data for task: {task}")
        
        # Generate vocabulary and cell sentences
        vocabulary = generate_vocabulary(adata)
        cell_sentences = generate_cell_sentences(
            adata, 
            vocabulary, 
            top_k_genes=self.config.top_k_genes,
            delimiter=self.config.sentence_delimiter
        )
        
        # Get cell type labels
        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs")
        
        cell_types = adata.obs[cell_type_col].tolist()
        
        # Create task-specific prompts
        if task == "cell_type_prediction":
            dataset = create_cell_type_prediction_prompts(cell_sentences, cell_types)
        elif task == "cell_generation":
            dataset = create_cell_generation_prompts(cell_sentences, cell_types)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Tokenize the dataset
        print("🔧 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_for_training(
                examples, 
                self.tokenizer, 
                self.config.loss_on_response_only
            ),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split the dataset
        dataset_size = len(tokenized_dataset)
        train_size = int(self.config.train_split * dataset_size)
        val_size = int(self.config.val_split * dataset_size)
        
        # Create splits
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        dataset_dict = DatasetDict({
            "train": tokenized_dataset.select(train_indices),
            "validation": tokenized_dataset.select(val_indices),
            "test": tokenized_dataset.select(test_indices) if test_indices else None
        })
        
        print(f"📈 Dataset splits - Train: {len(dataset_dict['train'])}, "
              f"Val: {len(dataset_dict['validation'])}, "
              f"Test: {len(dataset_dict['test']) if dataset_dict['test'] else 0}")
        
        return dataset_dict
    
    def train(self, dataset_dict: DatasetDict):
        """
        Train the Cell2Sentence model.
        
        Arguments:
            dataset_dict: DatasetDict containing train/validation splits
        """
        print("🚀 Starting Cell2Sentence training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_drop_last=True,
        )
        
        # Setup data collator
        data_collator = Cell2SentenceDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_position_embeddings
        )
        
        # Limit evaluation samples if specified
        eval_dataset = dataset_dict["validation"]
        if (self.config.max_eval_samples and 
            len(eval_dataset) > self.config.max_eval_samples):
            eval_indices = random.sample(
                range(len(eval_dataset)), 
                self.config.max_eval_samples
            )
            eval_dataset = eval_dataset.select(eval_indices)
            print(f"📊 Limited evaluation dataset to {len(eval_dataset)} samples")
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print("🏃‍♂️ Training started!")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"✅ Training completed! Model saved to: {self.config.output_dir}")
    
    def generate_text(self, 
                     prompt: str, 
                     max_new_tokens: int = 50,
                     temperature: float = 0.8,
                     do_sample: bool = True,
                     top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """
        Generate text using the trained model.
        
        Arguments:
            prompt: Input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode and clean output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, "").strip()
        
        return generated_text

# -----------------------------------------------
# Demo Functions
# -----------------------------------------------

def create_synthetic_data(n_cells: int = 1000, n_genes: int = 500) -> anndata.AnnData:
    """
    Create synthetic single-cell data for demonstration.
    
    Arguments:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate
        
    Returns:
        AnnData object with synthetic expression data
    """
    print(f"📊 Creating synthetic data: {n_cells} cells × {n_genes} genes")
    
    # Create synthetic expression matrix
    np.random.seed(42)
    expression_data = np.random.lognormal(0, 1, (n_cells, n_genes))
    expression_data = np.where(expression_data > 2, expression_data, 0)  # Add sparsity
    
    # Create gene names
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    
    # Create cell types
    cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte"] * (n_cells // 4)
    cell_types = cell_types[:n_cells]
    
    # Create AnnData object
    adata = anndata.AnnData(
        X=expression_data,
        obs={"cell_type": cell_types},
        var={"gene_names": gene_names}
    )
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    print(f"✅ Synthetic dataset created!")
    return adata

def load_real_adata(data_file: str) -> anndata.AnnData:
    """
    Load real AnnData file and prepare it for Cell2Sentence training.
    
    Arguments:
        data_file: Path to .h5ad file
        
    Returns:
        Loaded and validated AnnData object
    """
    print(f"📂 Loading real data from: {data_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load the data
    adata = sc.read_h5ad(data_file)
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"   🔬 Observations (cells): {list(adata.obs.columns)}")
    print(f"   🧬 Variables (genes): {adata.var.shape[0]} genes")
    
    # Check for common cell type columns
    potential_cell_type_cols = [
        'cell_type', 'celltype', 'cell_ontology_class', 
        'annotation', 'cluster', 'leiden', 'seurat_clusters'
    ]
    
    available_cell_type_cols = [col for col in potential_cell_type_cols if col in adata.obs.columns]
    
    if available_cell_type_cols:
        print(f"   🏷️  Available cell type columns: {available_cell_type_cols}")
        
        # Show unique values for the first available column
        first_col = available_cell_type_cols[0]
        unique_types = adata.obs[first_col].unique()
        print(f"   📝 Cell types in '{first_col}': {list(unique_types[:10])}{'...' if len(unique_types) > 10 else ''}")
        print(f"   📈 Total unique cell types: {len(unique_types)}")
    else:
        print("   ⚠️  No obvious cell type column found. Available columns:")
        for col in adata.obs.columns:
            print(f"      - {col}")
    
    return adata

def run_demo(model_path: str = None, tokenizer_path: str = None, train_model: bool = False, data_file: str = None):
    """
    Run the Cell2Sentence demonstration.
    
    Arguments:
        model_path: Path to local GPT-2 model (optional)
        tokenizer_path: Path to local GPT-2 tokenizer (optional)
        train_model: Whether to actually train the model (takes time!)
        data_file: Path to real AnnData file (.h5ad) (optional)
    """
    print("🧬 Cell2Sentence Demo")
    print("=" * 50)
    
    # Configuration
    config = Cell2SentenceConfig(
        model_name_or_path=model_path or "gpt2",
        output_dir="./cell2sentence_demo",
        num_train_epochs=1 if train_model else 1,  # Reduced for demo
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        top_k_genes=50,
        max_eval_samples=100,
        learning_rate=5e-5,
    )
    
    # Load data - real or synthetic
    if data_file:
        adata = load_real_adata(data_file)
        
        # Detect cell type column
        potential_cols = ['cell_type', 'celltype', 'cell_ontology_class', 'annotation', 'cluster', 'leiden']
        cell_type_col = None
        
        for col in potential_cols:
            if col in adata.obs.columns:
                cell_type_col = col
                print(f"🎯 Using '{col}' as cell type column")
                break
        
        if not cell_type_col:
            print("⚠️  No cell type column found, using first available column")
            cell_type_col = adata.obs.columns[0]
            
    else:
        adata = create_synthetic_data(n_cells=200, n_genes=100)  # Smaller for demo
        cell_type_col = "cell_type"
    
    # Initialize trainer
    trainer = Cell2SentenceTrainer(config, model_path, tokenizer_path)
    
    # Prepare data
    dataset_dict = trainer.prepare_data_from_adata(
        adata, 
        task="cell_type_prediction",
        cell_type_col=cell_type_col
    )
    
    # Train model (optional)
    if train_model:
        print("⚠️  Starting training - this may take a while...")
        trainer.train(dataset_dict)
    else:
        print("⏭️  Skipping training (set train_model=True to train)")
    
    # Test generation
    print("\n🔮 Testing Cell2Sentence generation...")
    test_prompts = [
        "Given the following gene expression pattern, predict the cell type:\nGenes: Gene_001 Gene_042 Gene_023\nCell type:",
        "What cell type does this gene expression pattern represent?\nGenes: Gene_010 Gene_005 Gene_030\nCell type:",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}:")
        print(f"Prompt: {prompt}")
        try:
            generated = trainer.generate_text(prompt, max_new_tokens=10)
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print("\n✅ Demo completed successfully!")

# -----------------------------------------------
# Main Execution
# -----------------------------------------------

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Cell2Sentence Training Demo")
    parser.add_argument("--model-path", type=str, help="Path to local GPT-2 model directory")
    parser.add_argument("--tokenizer-path", type=str, help="Path to local GPT-2 tokenizer directory")
    parser.add_argument("--train", action="store_true", help="Actually train the model (takes time!)")
    parser.add_argument("--data-file", type=str, help="Path to AnnData file (.h5ad)")
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    if not args.model_path and not args.tokenizer_path:
        print("ℹ️  No local model paths provided, will use online GPT-2")
        model_path = None
        tokenizer_path = None
    else:
        model_path = args.model_path
        tokenizer_path = args.tokenizer_path
        
        # Validate paths exist
        if model_path and not os.path.exists(model_path):
            print(f"❌ Model path does not exist: {model_path}")
            return
        if tokenizer_path and not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer path does not exist: {tokenizer_path}")
            return
    
    # Run the demo
    try:
        run_demo(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            train_model=args.train,
            data_file=args.data_file
        )
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
