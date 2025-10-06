"""
Train split model on full FinTabNet dataset
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from final_model import SplitModel, TableDataset, focal_loss
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_h_acc = 0
    total_v_acc = 0

    start_time = time.time()
    for batch_idx, (images, h_targets, v_targets) in enumerate(dataloader):
        images = images.to(device)
        h_targets = h_targets.to(device)
        v_targets = v_targets.to(device)

        # Forward
        h_pred, v_pred = model(images)

        # Loss
        h_loss = focal_loss(h_pred, h_targets)
        v_loss = focal_loss(v_pred, v_targets)
        loss = h_loss + v_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        h_binary = (h_pred > 0.5).float()
        v_binary = (v_pred > 0.5).float()
        h_acc = (h_binary == h_targets).float().mean().item()
        v_acc = (v_binary == v_targets).float().mean().item()
        total_h_acc += h_acc
        total_v_acc += v_acc

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                       f"Loss: {loss.item():.4f} | "
                       f"H_acc: {h_acc:.3f} | V_acc: {v_acc:.3f} | "
                       f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / len(dataloader)
    avg_h_acc = total_h_acc / len(dataloader)
    avg_v_acc = total_v_acc / len(dataloader)

    return avg_loss, avg_h_acc, avg_v_acc

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_h_acc = 0
    total_v_acc = 0

    with torch.no_grad():
        for images, h_targets, v_targets in dataloader:
            images = images.to(device)
            h_targets = h_targets.to(device)
            v_targets = v_targets.to(device)

            h_pred, v_pred = model(images)

            h_loss = focal_loss(h_pred, h_targets)
            v_loss = focal_loss(v_pred, v_targets)
            loss = h_loss + v_loss

            total_loss += loss.item()

            h_binary = (h_pred > 0.5).float()
            v_binary = (v_pred > 0.5).float()
            h_acc = (h_binary == h_targets).float().mean().item()
            v_acc = (v_binary == v_targets).float().mean().item()
            total_h_acc += h_acc
            total_v_acc += v_acc

    avg_loss = total_loss / len(dataloader)
    avg_h_acc = total_h_acc / len(dataloader)
    avg_v_acc = total_v_acc / len(dataloader)

    return avg_loss, avg_h_acc, avg_v_acc

def main():
    # Config
    batch_size = 32
    num_epochs = 16
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")

    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")
    train_dataset = TableDataset(ds['train'])
    val_dataset = TableDataset(ds['val'])

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # DataLoaders
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

    # Model
    model = SplitModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_h_acc, train_v_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_h_acc, val_v_acc = validate(model, val_loader, device)

        logger.info(f"  Train - Loss: {train_loss:.4f} | H_acc: {train_h_acc:.3f} | V_acc: {train_v_acc:.3f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f} | H_acc: {val_h_acc:.3f} | V_acc: {val_v_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_h_acc': val_h_acc,
                'val_v_acc': val_v_acc
            }
            torch.save(checkpoint, 'best_model_full.pth')
            logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

