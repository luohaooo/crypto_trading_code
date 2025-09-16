"""
OHLC Model Training Pipeline

This module provides a complete training pipeline for the OHLC deep learning model.
Includes training loops, validation, checkpointing, and early stopping.
"""

import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path

from ohlc_model import Net, create_model
from training_dataset import create_training_dataset, create_data_loaders
from typing import List, Dict, Optional, Tuple

import sys
project_root = '/home/craz/crypto/crypto-trading'
sys.path.insert(0, project_root)

# Import neural-strategy utils
neural_strategy_root = '/home/craz/crypto/crypto-trading/neural-strategy'
sys.path.insert(0, neural_strategy_root)

from utils.dingding import send_dingtalk_message


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """
    Training loop - fixed version

    Args:
        dataloader: Training data loader
        model: Neural network model
        loss_fn: Loss function
        optimizer: Optimizer
        device: Training device

    Returns:
        Average training loss
    """
    running_loss = 0.0
    current = 0
    model.train()

    with tqdm(dataloader) as t:
        for X, y in t:
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(X)

            # Ensure correct label type (float for regression)
            if loss_fn.__class__.__name__ in ['L1Loss', 'MSELoss']:
                y = y.float()
            else:
                y = y.long()

            loss = loss_fn(y_pred.squeeze(), y)  # Ensure dimension matching

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Correct loss calculation
            batch_loss = loss.item()
            running_loss = (len(X) * batch_loss + running_loss * current) / (len(X) + current)
            current += len(X)
            t.set_postfix({'running_loss': running_loss})

    return running_loss


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> float:
    """
    Validation loop - fixed version

    Args:
        dataloader: Validation data loader
        model: Neural network model
        loss_fn: Loss function
        device: Training device

    Returns:
        Average validation loss
    """
    running_loss = 0.0
    current = 0
    model.eval()

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for X, y in t:
                X = X.to(device)
                y = y.to(device)

                # Forward pass
                y_pred = model(X)

                # Ensure correct label type (float for regression)
                if loss_fn.__class__.__name__ in ['L1Loss', 'MSELoss']:
                    y = y.float()
                else:
                    y = y.long()

                loss = loss_fn(y_pred.squeeze(), y)  # Ensure dimension matching

                # Fixed: correct loss calculation (consistent with training)
                batch_loss = loss.item()
                running_loss = (len(X) * batch_loss + running_loss * current) / (len(X) + current)
                current += len(X)
                t.set_postfix({'running_loss': running_loss})

    return running_loss


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epochs: int = 100,
                early_stopping_epoch: int = 5,
                start_epoch: int = 0,
                save_dir: str = "./model_checkpoint") -> str:
    """
    Train model with checkpointing and early stopping

    Args:
        model: Neural network model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Training device
        epochs: Maximum number of epochs
        early_stopping_epoch: Early stopping patience
        start_epoch: Starting epoch (for resuming)
        save_dir: Directory to save checkpoints

    Returns:
        Path to best model checkpoint
    """
    # Record training and validation losses
    train_losses = []
    val_losses = []

    # Record best validation loss
    min_val_loss = float("inf")
    last_min_ind = -1

    # Model storage directory
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, start_time)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔄 Starting training...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Early stopping patience: {early_stopping_epoch}")
    print(f"  - Save directory: {save_dir}")
    print(f"  - Device: {device}")

    best_ckpt_path = None

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n===== Epoch {epoch} =====")

        # Training
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)

        # Validation
        val_loss = val_loop(val_dataloader, model, loss_fn, device)
        val_losses.append(val_loss)

        # Save current model checkpoint
        ckpt_name = f"baseline_epoch_{epoch}_train_{train_loss:.5f}_val_{val_loss:.5f}.pt"
        ckpt_path = os.path.join(save_dir, ckpt_name)

        # Save model state dict (works with both single model and DataParallel)
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)

        print(f"✔ Saved checkpoint: {ckpt_path}")

        # Early stopping logic
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            last_min_ind = epoch
            best_ckpt_path = ckpt_path
        elif epoch - last_min_ind >= early_stopping_epoch:
            print(f"⏹ Early stopping at epoch {epoch} (no improvement for {early_stopping_epoch} epochs)")
            break

    print("\n🎉 Training completed!")
    print(f"Best epoch: {last_min_ind}, best val_loss: {min_val_loss:.5f}")

    # Copy best model to final location
    if best_ckpt_path:
        final_model_dir = os.path.join(os.path.dirname(save_dir), "model_saved")
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = os.path.join(final_model_dir, os.path.basename(best_ckpt_path))

        import shutil
        shutil.copy2(best_ckpt_path, final_model_path)
        print(f"✔ Best model saved to: {final_model_path}")

        return final_model_path

    return best_ckpt_path


def create_training_pipeline(data_dir: str,
                           label_horizon: str = '1d',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           batch_size: int = 256,
                           learning_rate: float = 5e-5,
                           epochs: int = 100,
                           early_stopping: int = 5,
                           train_ratio: float = 0.7,
                           save_dir: str = "./model_checkpoint",
                           preload_all: bool = False) -> Tuple[str, Dict]:
    """
    Complete training pipeline

    Args:
        data_dir: Data directory path
        label_horizon: Return prediction horizon
        start_date: Start date for training data
        end_date: End date for training data
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Maximum epochs
        early_stopping: Early stopping patience
        train_ratio: Training data ratio
        save_dir: Model save directory
        preload_all: Whether to preload all data

    Returns:
        best_model_path, training_info
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Create dataset
    print(f"\n📊 Creating dataset...")
    dataset = create_training_dataset(
        data_dir=data_dir,
        label_horizon=label_horizon,
        start_date=start_date,
        end_date=end_date,
        preload_all=preload_all
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_ratio=train_ratio,
        batch_size=batch_size,
        shuffle=True
    )

    # Create model
    print(f"\n🧠 Creating model...")
    model = create_model(device=device, use_parallel=torch.cuda.device_count() > 1)
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else model.module.get_model_info()

    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # Setup training components
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n🎯 Training configuration:")
    print(f"  - Loss function: {loss_fn.__class__.__name__}")
    print(f"  - Optimizer: {optimizer.__class__.__name__}")
    print(f"  - Learning rate: {learning_rate}")

    # Train model
    start_time = time.time()
    best_model_path = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        early_stopping_epoch=early_stopping,
        save_dir=save_dir
    )
    training_time = time.time() - start_time

    training_info = {
        'dataset_info': dataset.get_data_info(),
        'model_info': model_info,
        'training_params': {
            'label_horizon': label_horizon,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'early_stopping': early_stopping,
            'train_ratio': train_ratio
        },
        'training_time': training_time,
        'best_model_path': best_model_path
    }

    print(f"\n✅ Training pipeline completed!")
    print(f"  - Training time: {training_time/60:.1f} minutes")
    print(f"  - Best model: {best_model_path}")

    return best_model_path, training_info


if __name__ == "__main__":
    # Example training pipeline
    DATA_DIR = '/home/craz/crypto/model_training/ohlc_img_dataset'

    print("🚀 Starting OHLC model training pipeline...")

    for [s, e] in [['2025-01','2025-04'], ['2024-12','2025-03'], ['2024-11','2025-02']]:

        try:
            best_model_path, training_info = create_training_pipeline(
                data_dir=DATA_DIR,
                label_horizon='1d',
                start_date=s,
                end_date=e,
                batch_size=256,
                learning_rate=5e-5,
                epochs=50,
                early_stopping=5,
                train_ratio=0.7,
                save_dir="./model_checkpoint",
                preload_all=False
            )

            print(f"\n📋 Training Summary:")
            print(f"  - Dataset: {training_info['dataset_info']['total_samples']:,} samples")
            print(f"  - Model: {training_info['model_info']['total_parameters']:,} parameters")
            print(f"  - Training time: {training_info['training_time']/60:.1f} minutes")
            print(f"  - Best model: {best_model_path}")

            send_dingtalk_message(f"start_date={s}, end_date={e}, Training time: {training_info['training_time']/60:.1f} minutes, Best model: {best_model_path}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
