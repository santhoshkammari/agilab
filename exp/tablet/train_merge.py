"""
Train merge model on full FinTabNet dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from merge_model import MergeModel, MergeDataset, focal_loss
import time
import logging
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Setup logging
def setup_logging(log_dir):
    """Setup comprehensive logging and tensorboard"""
    os.makedirs(log_dir, exist_ok=True)

    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create custom logger
    logger = logging.getLogger('TABLET_MERGE')
    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler with detailed format
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Setup tensorboard with metadata
    writer = SummaryWriter(log_dir, comment='_TABLET_Merge_Detection')

    # Log system information
    logger.info("=" * 60)
    logger.info("🚀 TABLET Merge Detection Training Started")
    logger.info("=" * 60)
    logger.info(f"📁 Log directory: {log_dir}")
    logger.info(f"🔧 PyTorch version: {torch.__version__}")
    logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU device: {torch.cuda.get_device_name()}")
        logger.info(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return writer, logger


def custom_collate_fn(batch):
    """
    Custom collate function for variable-sized grid boxes

    Args:
        batch: list of tuples (image, grid_boxes, grid_labels)

    Returns:
        images: [B, 3, 960, 960]
        grid_boxes: list of B tensors
        grid_labels: list of B tensors
    """
    images = torch.stack([item[0] for item in batch])
    grid_boxes = [item[1] for item in batch]
    grid_labels = [item[2] for item in batch]
    return images, grid_boxes, grid_labels


def train_epoch(model, dataloader, optimizer, device, epoch, logger, writer, global_step):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    epoch_start_time = datetime.now()

    for batch_idx, (images, grid_boxes, grid_labels) in enumerate(dataloader):
        images = images.to(device)
        grid_boxes = [boxes.to(device) for boxes in grid_boxes]
        grid_labels = [labels.to(device) for labels in grid_labels]

        # Forward
        optimizer.zero_grad()
        predictions = model(images, grid_boxes)

        # Calculate loss for each sample in batch
        batch_loss = 0
        batch_acc = 0
        valid_samples = 0

        for pred, labels in zip(predictions, grid_labels):
            if pred.size(0) > 0:  # Skip empty grids
                loss = focal_loss(pred, labels, alpha=1.0, gamma=2.0)
                batch_loss += loss

                # Calculate accuracy
                pred_classes = pred.argmax(dim=1)
                acc = (pred_classes == labels).float().mean()
                batch_acc += acc.item()
                valid_samples += 1

        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            batch_acc = batch_acc / valid_samples
        else:
            continue  # Skip batch if no valid samples

        # Backward
        batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # Track metrics
        total_loss += batch_loss.item()
        total_acc += batch_acc
        num_batches += 1

        # Tensorboard logging
        writer.add_scalar('Train/Loss_Step', batch_loss.item(), global_step)
        writer.add_scalar('Train/Accuracy_Step', batch_acc, global_step)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

        # Log gradient norms periodically
        if global_step % 100 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar('Train/Gradient_Norm', total_norm, global_step)

        global_step += 1

        # Log progress every batch
        if (batch_idx + 1) % 1 == 0:
            elapsed = datetime.now() - epoch_start_time
            progress = (batch_idx + 1) / len(dataloader) * 100
            log_msg = f'Epoch {epoch+1} [{progress:5.1f}%] Batch {batch_idx+1}/{len(dataloader)} | Loss: {batch_loss.item():.4f} | Acc: {batch_acc:.3f} | Elapsed: {elapsed}'
            logger.info(log_msg)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0

    return avg_loss, avg_acc, global_step


def validate(model, dataloader, device, logger):
    model.eval()
    total_loss = 0
    total_acc = 0

    # Per-class metrics
    class_correct = [0, 0, 0, 0]  # C, L, U, X
    class_total = [0, 0, 0, 0]

    num_batches = 0

    logger.info("🔍 Starting validation evaluation...")

    with torch.no_grad():
        for batch_idx, (images, grid_boxes, grid_labels) in enumerate(dataloader):
            images = images.to(device)
            grid_boxes = [boxes.to(device) for boxes in grid_boxes]
            grid_labels = [labels.to(device) for labels in grid_labels]

            predictions = model(images, grid_boxes)

            batch_loss = 0
            batch_acc = 0
            valid_samples = 0

            for pred, labels in zip(predictions, grid_labels):
                if pred.size(0) > 0:
                    loss = focal_loss(pred, labels, alpha=1.0, gamma=2.0)
                    batch_loss += loss

                    pred_classes = pred.argmax(dim=1)
                    acc = (pred_classes == labels).float().mean()
                    batch_acc += acc.item()
                    valid_samples += 1

                    # Per-class accuracy
                    for c in range(4):
                        mask = (labels == c)
                        class_total[c] += mask.sum().item()
                        class_correct[c] += ((pred_classes == labels) & mask).sum().item()

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_acc = batch_acc / valid_samples

                total_loss += batch_loss.item()
                total_acc += batch_acc
                num_batches += 1

            # Log progress every 20 batches
            if logger and batch_idx % 20 == 0:
                logger.debug(f"Validation batch {batch_idx}/{len(dataloader)}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0

    # Calculate per-class accuracy
    class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                 for i in range(4)]

    logger.info("✅ Validation completed")

    metrics = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'class_acc_C': class_acc[0],
        'class_acc_L': class_acc[1],
        'class_acc_U': class_acc[2],
        'class_acc_X': class_acc[3]
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train TABLET Merge Detection Model')

    # Dataset arguments
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Number of images to train on (-1 for full dataset)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=16,
                        help='Number of epochs to train (default: 16)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    # Model arguments
    parser.add_argument('--split-model-path', type=str, default=None,
                        help='Path to trained split model (optional, uses ground truth if not provided)')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging and tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/tablet_merge_{timestamp}"
    writer, logger = setup_logging(log_dir)

    logger.info(f"🎮 Using device: {device}")

    # Load dataset
    logger.info("📥 Loading FinTabNet dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Load split model if provided
    split_model = None
    if args.split_model_path:
        logger.info(f"📥 Loading split model from {args.split_model_path}")
        from final_model import SplitModel
        split_model = SplitModel().to(device)
        checkpoint = torch.load(args.split_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            split_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            split_model.load_state_dict(checkpoint)
        split_model.eval()
        logger.info("✅ Split model loaded successfully")

    # Create datasets
    train_dataset = MergeDataset(ds['train'], split_model=split_model, device=device)
    val_dataset = MergeDataset(ds['val'], split_model=split_model, device=device)

    # Subset for testing if specified
    if args.num_images > 0:
        logger.info(f"🔬 Using subset of {args.num_images} images for training")
        train_indices = list(range(min(args.num_images, len(train_dataset))))
        val_indices = list(range(min(args.num_images // 5, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    logger.info(f"📊 Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # Initialize model
    model = MergeModel(max_grid_cells=640).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

    # Log to tensorboard
    writer.add_scalar('Dataset/Train_Size', len(train_dataset), 0)
    writer.add_scalar('Dataset/Val_Size', len(val_dataset), 0)
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-4
    )

    # Log training configuration
    logger.info(f"📋 Training Configuration:")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ Batch Size: {args.batch_size}")
    logger.info(f"   ├─ Learning Rate: {args.lr}")
    logger.info(f"   ├─ Weight Decay: 5e-4")
    logger.info(f"   └─ Gradient Clipping: 0.5")

    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    global_step = 0
    start_time = datetime.now()

    logger.info("🚀 Starting training loop...")

    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        logger.info(f"📖 Epoch {epoch+1}/{args.epochs} started")

        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, logger, writer, global_step
        )

        # Validate
        epoch_duration = datetime.now() - epoch_start_time
        logger.info(f"🗓️ Epoch {epoch+1} training completed in {epoch_duration}")
        val_metrics = validate(model, val_loader, device, logger)

        # Comprehensive epoch logging
        logger.info(f"📊 Epoch {epoch+1}/{args.epochs} Results:")
        logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        logger.info(f"   Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")
        logger.info(f"   Val Class Acc - C: {val_metrics['class_acc_C']:.3f}, L: {val_metrics['class_acc_L']:.3f}, U: {val_metrics['class_acc_U']:.3f}, X: {val_metrics['class_acc_X']:.3f}")

        # Tensorboard logging
        writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
        writer.add_scalar('Train/Accuracy_Epoch', train_acc, epoch)
        writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Accuracy_Epoch', val_metrics['accuracy'], epoch)
        writer.add_scalar('Val/Class_Acc_C', val_metrics['class_acc_C'], epoch)
        writer.add_scalar('Val/Class_Acc_L', val_metrics['class_acc_L'], epoch)
        writer.add_scalar('Val/Class_Acc_U', val_metrics['class_acc_U'], epoch)
        writer.add_scalar('Val/Class_Acc_X', val_metrics['class_acc_X'], epoch)

        # Save best model (based on accuracy)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(log_dir, 'best_merge_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
            }, best_model_path)
            logger.info(f'✨ New best model saved! Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f}')

        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)
            logger.info(f'💾 Checkpoint saved: {checkpoint_path}')

    # Training completion
    total_training_time = datetime.now() - start_time
    logger.info("="*60)
    logger.info("🏆 Training Completed Successfully!")
    logger.info(f"⏱️ Total training time: {total_training_time}")
    logger.info(f"🎆 Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"🎆 Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)

    # Save final model
    final_model_path = os.path.join(log_dir, 'final_merge_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'💾 Final model saved: {final_model_path}')

    # Close tensorboard writer
    writer.close()
    logger.info("📈 Tensorboard logs closed")

    return model, log_dir


if __name__ == "__main__":
    main()

