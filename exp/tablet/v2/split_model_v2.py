"""
Fixed training script for TABLET split model with proper loss handling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from split_model import SplitModel, TableDataset,focal_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config (following TABLET paper specifications)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32  # Paper specification
epochs = 16  # Paper specification
lr = 3e-4  # Paper: 3e-4
pos_weight = 15.0  # Weight positive class heavily (not used with focal loss)
split_width = 5  # Width of split regions in pixels (hyperparameter)

# Load dataset
logger.info("Loading dataset...")
ds = load_dataset("ds4sd/FinTabNet_OTSL")

# Use full dataset for better performance
original_train_dataset = ds['train'].select(range(1000))
original_val_dataset = ds['val'].select(range(1000))

train_dataset = TableDataset(original_train_dataset, split_width=split_width)
val_dataset = TableDataset(original_val_dataset, split_width=split_width)

logger.info(f"📊 Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
logger.info(f"📊 Split width: {split_width} pixels")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Initialize model
model = SplitModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"📊 Model: {total_params:,} params ({trainable_params:,} trainable)")

# Optimizer with weight decay (paper specs: lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=5e-4)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=5e-4
)

# Learning rate scheduler - reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# TensorBoard
writer = SummaryWriter('runs/split_model_fixed')

# Check initial predictions
logger.info("Checking untrained model...")
model.eval()
with torch.no_grad():
    test_batch = next(iter(train_loader))
    test_images, test_h_gt, test_v_gt = test_batch
    test_images = test_images.to(device)
    test_h_pred, test_v_pred = model(test_images)

    logger.info(f"Untrained: H_mean={test_h_pred.mean():.4f}, V_mean={test_v_pred.mean():.4f}")
    logger.info(f"Untrained: H>0.5={(test_h_pred>0.5).float().mean()*100:.2f}%, V>0.5={(test_v_pred>0.5).float().mean()*100:.2f}%")


def compute_metrics(pred, target):
    """Compute Precision, Recall, F1 for split detection"""
    pred_binary = (pred > 0.5).float()

    # True positives, false positives, false negatives
    tp = (pred_binary * target).sum().item()
    fp = (pred_binary * (1 - target)).sum().item()
    fn = ((1 - pred_binary) * target).sum().item()

    # Precision: Of all predicted splits, how many are correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: Of all ground truth splits, how many did we find?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1: Harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Count error: absolute difference in number of splits
    count_error = abs(pred_binary.sum().item() - target.sum().item())

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count_error': count_error
    }


# Training loop
logger.info("🚀 Starting training...")
global_step = 0
best_val_f1 = 0.0  # Track best validation F1 score

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_h_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
    epoch_v_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, h_targets, v_targets) in enumerate(pbar):
        images = images.to(device)
        h_targets = h_targets.to(device)
        v_targets = v_targets.to(device)

        optimizer.zero_grad()

        # Forward
        h_pred, v_pred = model(images)

        # Weighted BCE loss
        # Convert sigmoid outputs back to logits for BCE with logits
        # h_logits = torch.logit(h_pred.clamp(1e-7, 1-1e-7))
        # v_logits = torch.logit(v_pred.clamp(1e-7, 1-1e-7))

        # pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        # h_loss = F.binary_cross_entropy_with_logits(h_logits, h_targets, pos_weight=pos_weight_tensor)
        # v_loss = F.binary_cross_entropy_with_logits(v_logits, v_targets, pos_weight=pos_weight_tensor)
        # loss = h_loss + v_loss

        h_loss = focal_loss(h_pred,h_targets)
        v_loss = focal_loss(v_pred,v_targets)
        loss = h_loss + v_loss


        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Paper: L2 norm with max_norm=0.5
        optimizer.step()

        # Metrics
        h_metrics = compute_metrics(h_pred, h_targets)
        v_metrics = compute_metrics(v_pred, v_targets)

        # Accumulate
        epoch_loss += loss.item()
        for k in epoch_h_metrics:
            epoch_h_metrics[k] += h_metrics[k]
            epoch_v_metrics[k] += v_metrics[k]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'H_F1': f'{h_metrics["f1"]:.2f}',
            'V_F1': f'{v_metrics["f1"]:.2f}'
        })

        # TensorBoard logging
        if batch_idx % 100 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/H_F1', h_metrics['f1'], global_step)
            writer.add_scalar('Train/V_F1', v_metrics['f1'], global_step)
            writer.add_scalar('Train/H_Precision', h_metrics['precision'], global_step)
            writer.add_scalar('Train/V_Precision', v_metrics['precision'], global_step)
            writer.add_scalar('Train/H_Recall', h_metrics['recall'], global_step)
            writer.add_scalar('Train/V_Recall', v_metrics['recall'], global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    # Epoch averages
    avg_loss = epoch_loss / num_batches
    for k in epoch_h_metrics:
        epoch_h_metrics[k] /= num_batches
        epoch_v_metrics[k] /= num_batches

    logger.info(f"\n📊 Epoch {epoch+1} Results:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  H - F1: {epoch_h_metrics['f1']:.3f}, Precision: {epoch_h_metrics['precision']:.3f}, Recall: {epoch_h_metrics['recall']:.3f}")
    logger.info(f"  V - F1: {epoch_v_metrics['f1']:.3f}, Precision: {epoch_v_metrics['precision']:.3f}, Recall: {epoch_v_metrics['recall']:.3f}")

    # Validation
    model.eval()
    val_loss = 0
    val_h_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
    val_v_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
    val_batches = 0

    with torch.no_grad():
        for images, h_targets, v_targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            h_targets = h_targets.to(device)
            v_targets = v_targets.to(device)

            h_pred, v_pred = model(images)

            # Loss
            # h_logits = torch.logit(h_pred.clamp(1e-7, 1-1e-7))
            # v_logits = torch.logit(v_pred.clamp(1e-7, 1-1e-7))
            # pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            # h_loss = F.binary_cross_entropy_with_logits(h_logits, h_targets, pos_weight=pos_weight_tensor)
            # v_loss = F.binary_cross_entropy_with_logits(v_logits, v_targets, pos_weight=pos_weight_tensor)
            # val_loss += (h_loss + v_loss).item()

            h_loss = focal_loss(h_pred,h_targets)
            v_loss = focal_loss(v_pred,v_targets)
            val_loss += (h_loss + v_loss).item()

            # Metrics
            h_m = compute_metrics(h_pred, h_targets)
            v_m = compute_metrics(v_pred, v_targets)
            for k in ['precision', 'recall', 'f1', 'count_error']:
                val_h_metrics[k] += h_m[k]
                val_v_metrics[k] += v_m[k]
            val_batches += 1

    avg_val_loss = val_loss / val_batches
    for k in val_h_metrics:
        val_h_metrics[k] /= val_batches
        val_v_metrics[k] /= val_batches

    logger.info(f"  Val Loss: {avg_val_loss:.4f}")
    logger.info(f"  Val H - F1: {val_h_metrics['f1']:.3f}, Precision: {val_h_metrics['precision']:.3f}, Recall: {val_h_metrics['recall']:.3f}")
    logger.info(f"  Val V - F1: {val_v_metrics['f1']:.3f}, Precision: {val_v_metrics['precision']:.3f}, Recall: {val_v_metrics['recall']:.3f}")

    # TensorBoard epoch metrics
    writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
    writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
    writer.add_scalar('Epoch/Val_H_F1', val_h_metrics['f1'], epoch)
    writer.add_scalar('Epoch/Val_V_F1', val_v_metrics['f1'], epoch)
    writer.add_scalar('Epoch/Val_H_Precision', val_h_metrics['precision'], epoch)
    writer.add_scalar('Epoch/Val_V_Precision', val_v_metrics['precision'], epoch)
    writer.add_scalar('Epoch/Val_H_Recall', val_h_metrics['recall'], epoch)
    writer.add_scalar('Epoch/Val_V_Recall', val_v_metrics['recall'], epoch)
    writer.add_scalar('Epoch/Val_H_Count_Error', val_h_metrics['count_error'], epoch)
    writer.add_scalar('Epoch/Val_V_Count_Error', val_v_metrics['count_error'], epoch)
    writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    # LR scheduler
    scheduler.step(avg_val_loss)

    # Save best model based on average validation F1 score
    avg_val_f1 = (val_h_metrics['f1'] + val_v_metrics['f1']) / 2
    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_h_f1': val_h_metrics['f1'],
            'val_v_f1': val_v_metrics['f1'],
            'val_h_precision': val_h_metrics['precision'],
            'val_v_precision': val_v_metrics['precision'],
            'val_h_recall': val_h_metrics['recall'],
            'val_v_recall': val_v_metrics['recall'],
        }, 'best_split_model.pth')
        logger.info(f"✨ New best model! Val F1: {avg_val_f1:.3f} (H: {val_h_metrics['f1']:.3f}, V: {val_v_metrics['f1']:.3f})")

    # Save periodic checkpoint
    if (epoch + 1) % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')
        logger.info(f"💾 Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

writer.close()
logger.info("✅ Training complete!")

