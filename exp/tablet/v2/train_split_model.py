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

# Load dataset
logger.info("Loading dataset...")
ds = load_dataset("ds4sd/FinTabNet_OTSL")

train_dataset = TableDataset(ds['train'].select(range(1000)))
val_dataset = TableDataset(ds['val'].select(range(1000)))

logger.info(f"📊 Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

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
    """Compute accuracy for positive and negative classes"""
    pred_binary = (pred > 0.5).float()

    # Overall accuracy
    acc = (pred_binary == target).float().mean().item()

    # Positive class accuracy (recall for splits)
    pos_mask = (target == 1)
    if pos_mask.sum() > 0:
        pos_acc = (pred_binary[pos_mask] == 1).float().mean().item()
    else:
        pos_acc = 0.0

    # Negative class accuracy
    neg_mask = (target == 0)
    if neg_mask.sum() > 0:
        neg_acc = (pred_binary[neg_mask] == 0).float().mean().item()
    else:
        neg_acc = 0.0

    return {
        'acc': acc,
        'pos_acc': pos_acc,
        'neg_acc': neg_acc
    }


# Training loop
logger.info("🚀 Starting training...")
global_step = 0
best_val_acc = 0.0  # Track best validation accuracy

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_h_metrics = {'acc': 0, 'pos_acc': 0, 'neg_acc': 0}
    epoch_v_metrics = {'acc': 0, 'pos_acc': 0, 'neg_acc': 0}
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
            'H_pos': f'{h_metrics["pos_acc"]:.2f}',
            'V_pos': f'{v_metrics["pos_acc"]:.2f}'
        })

        # TensorBoard logging
        if batch_idx % 100 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/H_Pos_Acc', h_metrics['pos_acc'], global_step)
            writer.add_scalar('Train/V_Pos_Acc', v_metrics['pos_acc'], global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    # Epoch averages
    avg_loss = epoch_loss / num_batches
    for k in epoch_h_metrics:
        epoch_h_metrics[k] /= num_batches
        epoch_v_metrics[k] /= num_batches

    logger.info(f"\n📊 Epoch {epoch+1} Results:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  H - Acc: {epoch_h_metrics['acc']:.3f}, Pos: {epoch_h_metrics['pos_acc']:.3f}, Neg: {epoch_h_metrics['neg_acc']:.3f}")
    logger.info(f"  V - Acc: {epoch_v_metrics['acc']:.3f}, Pos: {epoch_v_metrics['pos_acc']:.3f}, Neg: {epoch_v_metrics['neg_acc']:.3f}")

    # Validation
    model.eval()
    val_loss = 0
    val_h_metrics = {'acc': 0, 'pos_acc': 0, 'neg_acc': 0}
    val_v_metrics = {'acc': 0, 'pos_acc': 0, 'neg_acc': 0}
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
            for k in ['acc', 'pos_acc', 'neg_acc']:
                val_h_metrics[k] += h_m[k]
                val_v_metrics[k] += v_m[k]
            val_batches += 1

    avg_val_loss = val_loss / val_batches
    for k in val_h_metrics:
        val_h_metrics[k] /= val_batches
        val_v_metrics[k] /= val_batches

    logger.info(f"  Val Loss: {avg_val_loss:.4f}")
    logger.info(f"  Val H - Acc: {val_h_metrics['acc']:.3f}, Pos: {val_h_metrics['pos_acc']:.3f}, Neg: {val_h_metrics['neg_acc']:.3f}")
    logger.info(f"  Val V - Acc: {val_v_metrics['acc']:.3f}, Pos: {val_v_metrics['pos_acc']:.3f}, Neg: {val_v_metrics['neg_acc']:.3f}")

    # TensorBoard epoch metrics
    writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
    writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
    writer.add_scalar('Epoch/Val_H_Acc', val_h_metrics['acc'], epoch)
    writer.add_scalar('Epoch/Val_V_Acc', val_v_metrics['acc'], epoch)
    writer.add_scalar('Epoch/Val_H_Pos_Acc', val_h_metrics['pos_acc'], epoch)
    writer.add_scalar('Epoch/Val_V_Pos_Acc', val_v_metrics['pos_acc'], epoch)
    writer.add_scalar('Epoch/Val_H_Neg_Acc', val_h_metrics['neg_acc'], epoch)
    writer.add_scalar('Epoch/Val_V_Neg_Acc', val_v_metrics['neg_acc'], epoch)
    writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    # LR scheduler
    scheduler.step(avg_val_loss)

    # Save best model based on average validation accuracy
    avg_val_acc = (val_h_metrics['acc'] + val_v_metrics['acc']) / 2
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_h_acc': val_h_metrics['acc'],
            'val_v_acc': val_v_metrics['acc'],
            'val_h_pos_acc': val_h_metrics['pos_acc'],
            'val_v_pos_acc': val_v_metrics['pos_acc'],
        }, 'best_split_model.pth')
        logger.info(f"✨ New best model! Val acc: {avg_val_acc:.3f} (H: {val_h_metrics['acc']:.3f}, V: {val_v_metrics['acc']:.3f})")

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

