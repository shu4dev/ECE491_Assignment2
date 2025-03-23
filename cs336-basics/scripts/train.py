#!/usr/bin/env python3
"""
Train a text classifier on one or multiple GPUs.

This script trains a transformer-based model to classify text documents into 
binary categories (0 or 1) using the labels from labels.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from cs336_basics.model import TextClassifier
from cs336_basics.optimizer import get_cosine_lr
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, data_files, labels, tokenizer=None, max_length=512):
        """
        Args:
            data_files: List of paths to text files
            labels: List of labels corresponding to each file (0 or 1)
            tokenizer: Tokenizer to use for encoding text (if None, simple split is used)
            max_length: Maximum sequence length
        """
        self.data_files = data_files
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Read the text file
        with open(self.data_files[idx], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple tokenization if no tokenizer provided
        if self.tokenizer is None:
            # Split by whitespace and convert to integers (simple vocab mapping)
            tokens = [ord(c) % 10000 for c in text]  # Simple character-based encoding
            # Truncate or pad to max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            # Use the provided tokenizer
            tokens = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)
            tokens = tokens['input_ids']
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_dataset_from_csv(csv_path, split_ratio=0.8, seed=42):
    """
    Load dataset from a CSV file with file paths and labels.
    
    Args:
        csv_path: Path to CSV file with columns 'file_path' and 'label'
        split_ratio: Ratio of training data (remaining is used for validation)
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset
    """
    df = pd.read_csv(csv_path)
    
    # Make sure all file paths exist
    valid_files = []
    valid_labels = []
    for i, row in df.iterrows():
        if os.path.exists(row['file_path']):
            valid_files.append(row['file_path'])
            valid_labels.append(row['label'])
        else:
            logger.warning(f"File not found: {row['file_path']}")
    
    # Create dataset
    dataset = TextClassificationDataset(valid_files, valid_labels)
    
    # Split into train and validation
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


def train(
    labels_csv,
    output_dir,
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    num_classes,
    attn_pdrop,
    residual_pdrop,
    batch_size,
    train_steps,
    eval_interval,
    learning_rate,
    lr_scheduler,
    warmup_ratio,
    weight_decay,
    adam_beta1,
    adam_beta2,
    adam_eps,
    grad_clip,
    device,
    compile,
    dtype,
    wandb_project,
):
    # Load and split the dataset
    train_dataset, val_dataset = load_dataset_from_csv(labels_csv)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize the model
    model = TextClassifier(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        num_classes=num_classes,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
    )

    # Multi-GPU setup using DDP if available
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        seed = ddp_rank  # each process gets a different seed
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
    else:
        seed = 42
        ddp_world_size = 1
        is_master_process = True

    # Seed for reproducibility
    torch.manual_seed(seed)

    # Save the model config
    if is_master_process:
        model_config_output_path = os.path.join(output_dir, "model_config.json")
        logger.info(f"Saving model config to {model_config_output_path}")
        with open(model_config_output_path, "w") as f:
            json.dump(model.config, f, indent=4)

    # Setup dtype for training
    device_type = "cuda" if "cuda" in device else "cpu"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    if is_master_process:
        logger.info(f"Using dtype: {torch_dtype}")
    amp_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)
    )
    # GradScaler is only used for FP16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Move model to the device
    model = model.to(device)

    # Compile the model if requested and available
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set up the AdamW optimizer with weight decay on appropriate parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    # Training loop
    global_step = 0
    best_val_acc = 0.0
    
    # For early stopping
    patience = 5
    no_improvement_count = 0
    
    train_iterator = iter(train_loader)
    
    for step in tqdm(range(train_steps)):
        # Update learning rate if using cosine scheduler
        if lr_scheduler.lower() == "cosine":
            lr = get_cosine_lr(
                step,
                max_learning_rate=learning_rate,
                min_learning_rate=learning_rate * 0.1,
                warmup_iters=int(train_steps * warmup_ratio),
                cosine_cycle_iters=train_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = learning_rate
            
        # Get the next batch (with wraparound if needed)
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Forward and backward pass
        with amp_ctx:
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            
        scaler.scale(loss).backward()
        
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        global_step += 1
        
        # Log training metrics
        if is_master_process:
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            logger.info(f"Step {global_step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
            
            if wandb_project:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_accuracy": accuracy,
                    "learning_rate": lr
                }, step=global_step)
        
        # Evaluate on validation set
        if global_step % eval_interval == 0 and is_master_process:
            val_loss, val_accuracy = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                amp_ctx=amp_ctx
            )
            
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            if wandb_project:
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=global_step)
                
            # Save the best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                no_improvement_count = 0
                
                # Save model weights
                model_weights_output_path = os.path.join(output_dir, "best_model.pt")
                logger.info(f"Saving best model (accuracy: {val_accuracy:.4f}) to {model_weights_output_path}")
                
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), model_weights_output_path)
                else:
                    torch.save(model.state_dict(), model_weights_output_path)
            else:
                no_improvement_count += 1
                
            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f"Early stopping after {patience} evaluations without improvement")
                break
    
    # Final evaluation
    if is_master_process:
        val_loss, val_accuracy = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            amp_ctx=amp_ctx
        )
        
        logger.info(f"Final validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        
        if wandb_project:
            wandb.log({
                "final_val_loss": val_loss,
                "final_val_accuracy": val_accuracy
            }, step=global_step)
        
        # Save the final model
        model_weights_output_path = os.path.join(output_dir, "model.pt")
        logger.info(f"Saving final model to {model_weights_output_path}")
        
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), model_weights_output_path)
        else:
            torch.save(model.state_dict(), model_weights_output_path)
    
    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def evaluate(model, val_loader, device, amp_ctx):
    """Evaluate the model on the validation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in val_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        with amp_ctx:
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    model.train()
    return total_loss / total, correct / total


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@'
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        help="Path to CSV file with file paths and labels",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to write model configuration and trained model checkpoint",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Size of the vocabulary",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Maximum context length for the model",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="The dimensionality of the model embeddings and sublayer outputs.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="The number of Transformer layers to use.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of heads to use in multi-headed attention.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,
        help="Dimensionality of the feed-forward inner layer.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classification categories.",
    )
    parser.add_argument(
        "--attn-pdrop",
        type=float,
        default=0.1,
        help="If given, drop-out the attention probabilities with this rate.",
    )
    parser.add_argument(
        "--residual-pdrop",
        type=float,
        default=0.1,
        help="If given, apply dropout to output of each sub-layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size to use during training.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=10000,
        help="Number of training steps to perform",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluate on validation set every this many steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate to use during training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine"],
        default="cosine",
        help="Learning rate scheduler to use during training.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of total steps to use for LR warmup",
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.01, 
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Value to use for Adam beta_1",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="Value to use for Adam beta_2",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Value to use for Adam epsilon",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="If set, clip gradient norms to this value",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If true, compile the model with torch.compile",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        help="dtype to use when training",
    )
    parser.add_argument(
        '--wandb-project', 
        default="Transformer",
        type=str,
        help="If set, log results to the specified wandb project",
    )
    
    args = parser.parse_args()

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    # Rank 0 does logging, file creation, etc.
    is_master_process = int(os.environ["RANK"]) == 0 if is_ddp else True

    if is_master_process:
        logger.info("running %s", " ".join(sys.argv))

        # Make the directory for output if it doesn't already exist
        if os.path.exists(os.path.join(args.output_dir, "model.pt")):
            raise ValueError(
                f"output directory {args.output_dir} already exists and contains model.pt"
            )
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.wandb_project:
            wandb.login()
            wandb.init(
                # Set the project where this run will be logged
                project=args.wandb_project,
                config=vars(args),
                name=pathlib.Path(args.output_dir).name,
            )
    train(
        args.labels_csv,
        args.output_dir,
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.num_classes,
        args.attn_pdrop,
        args.residual_pdrop,
        args.batch_size,
        args.train_steps,
        args.eval_interval,
        args.learning_rate,
        args.lr_scheduler,
        args.warmup_ratio,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.grad_clip,
        args.device,
        args.compile,
        args.dtype,
        args.wandb_project,
    )
    logger.info("finished running %s", sys.argv[0])
