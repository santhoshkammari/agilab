import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from bs4 import BeautifulSoup
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

def get_ground_truth(image, html_tags, cells):
    """
    Generate ground truth for split lines

    Args:
        image: PIL Image object
        html_tags: list of HTML tags from dataset
        cells: list of cell data with tokens and bboxes from dataset

    Returns:
        horizontal_gt: list of 960 binary values (0/1) for row splits
        vertical_gt: list of 960 binary values (0/1) for column splits
    """
    # Convert HTML tags list to HTML string
    html_content = ''.join(html_tags)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Convert cells to OCR bbox format: (text, x1, y1, x2, y2)
    ocr_bboxes = []
    for cell_group in cells:
        for cell in cell_group:
            tokens = cell['tokens']
            bbox = cell['bbox']
            text = ''.join(tokens)
            # bbox format: [x1, y1, x2, y2, page] -> (text, x1, y1, x2, y2)
            ocr_bboxes.append((text, bbox[0], bbox[1], bbox[2], bbox[3]))
    rows = soup.find_all('tr')
    
    # Initialize ground truth arrays
    horizontal_gt = [0] * 960
    vertical_gt = [0] * 960
    
    # Sort OCR boxes by y-coordinate for row processing
    ocr_boxes = sorted(ocr_bboxes, key=lambda x: x[2])
    
    # Find row boundaries
    row_groups = []
    current_y = 0
    
    # Group OCR boxes by rows (boxes with similar y-coordinates)
    for box in ocr_boxes:
        if not row_groups or abs(box[2] - current_y) > 20:  # New row if y diff > 20px
            row_groups.append([])
            current_y = box[2]
        row_groups[-1].append(box)
    
    # Create horizontal split regions between rows
    for i in range(len(row_groups) - 1):
        curr_row_bottom = max([box[4] for box in row_groups[i]])    # Max y2 of current row
        next_row_top = min([box[2] for box in row_groups[i + 1]])   # Min y1 of next row
        
        # Split region is between rows
        split_start = curr_row_bottom
        split_end = next_row_top
        split_center = (split_start + split_end) // 2
        
        # Ensure minimum 5 pixel width as mentioned in paper
        split_width = max(5, split_end - split_start)
        split_start = split_center - split_width // 2
        split_end = split_center + split_width // 2
        
        # Mark split region in ground truth
        for y in range(max(0, split_start), min(960, split_end)):
            horizontal_gt[y] = 1
    
    # Find column boundaries using first row
    if row_groups:
        first_row_boxes = sorted(row_groups[0], key=lambda x: x[1])  # Sort by x-coordinate
        
        for i in range(len(first_row_boxes) - 1):
            curr_cell_right = first_row_boxes[i][3]   # x2 of current cell
            next_cell_left = first_row_boxes[i + 1][1] # x1 of next cell
            
            split_start = curr_cell_right
            split_end = next_cell_left
            split_center = (split_start + split_end) // 2
            
            # Ensure minimum 5 pixel width
            split_width = max(5, split_end - split_start)
            split_start = split_center - split_width // 2
            split_end = split_center + split_width // 2
            
            # Mark split region in ground truth
            for x in range(max(0, split_start), min(960, split_end)):
                vertical_gt[x] = 1
    
    return horizontal_gt, vertical_gt

class BasicBlock(nn.Module):
    """Basic ResNet block with halved channels"""
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class ModifiedResNet18(nn.Module):
    """ResNet-18 with removed maxpool and halved channels"""
    def __init__(self):
        super().__init__()
        # First conv block - halved channels: 64→32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # Skip maxpool - this is the removal mentioned in paper
        
        # ResNet layers with halved channels
        self.layer1 = self._make_layer(32, 32, 2, stride=1)    # Original: 64
        self.layer2 = self._make_layer(32, 64, 2, stride=2)    # Original: 128
        self.layer3 = self._make_layer(64, 128, 2, stride=2)   # Original: 256
        self.layer4 = self._make_layer(128, 256, 2, stride=2)  # Original: 512
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)    # [B, 32, 480, 480]
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool here - this is the key modification
        
        x = self.layer1(x)   # [B, 32, 480, 480]
        x = self.layer2(x)   # [B, 64, 240, 240]
        x = self.layer3(x)   # [B, 128, 120, 120]
        x = self.layer4(x)   # [B, 256, 60, 60]
        return x

class FPN(nn.Module):
    """Feature Pyramid Network outputting 128 channels at H/2×W/2"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 128, kernel_size=1)
        
    def forward(self, x):
        # x is [B, 256, 60, 60] from ResNet
        x = self.conv(x)  # [B, 128, 60, 60]
        # Upsample to H/2×W/2 = 480×480
        x = F.interpolate(x, size=(480, 480), mode='bilinear', align_corners=False)
        return x  # [B, 128, 480, 480]

class SplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ModifiedResNet18()
        self.fpn = FPN()
        
        # Learnable weights for global feature averaging
        self.h_global_weight = nn.Parameter(torch.randn(480))  # For width dimension
        self.v_global_weight = nn.Parameter(torch.randn(480))  # For height dimension
        
        # Local feature processing - reduce to 1 channel then treat spatial as features
        self.h_local_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.v_local_conv = nn.Conv2d(128, 1, kernel_size=1)
        
        # Positional embeddings (1D as mentioned in paper)
        self.h_pos_embed = nn.Parameter(torch.randn(480, 368))
        self.v_pos_embed = nn.Parameter(torch.randn(480, 368))

        # Transformers with correct dimensions
        self.h_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=368, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )
        self.v_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=368, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )

        # Classification heads
        self.h_classifier = nn.Linear(368, 1)
        self.v_classifier = nn.Linear(368, 1)
        
    def forward(self, x):
        # Input: [B, 3, 960, 960]
        features = self.backbone(x)      # [B, 256, 60, 60]
        F_half = self.fpn(features)      # [B, 128, 480, 480] - This is F1/2
        
        B, C, H, W = F_half.shape        # B, 128, 480, 480
        
        # HORIZONTAL FEATURES (for row splitting)
        # Global: learnable weighted average along width dimension
        F_RG = torch.einsum('bchw,w->bch', F_half, self.h_global_weight)  # [B, 128, 480]
        F_RG = F_RG.transpose(1, 2)  # [B, 480, 128]
        
        # Local: 1×2 AvgPool then 1×1 conv to 1 channel, treat spatial as features
        F_RL_pooled = F.avg_pool2d(F_half, kernel_size=(1, 2))  # [B, 128, 480, 240]
        F_RL = self.h_local_conv(F_RL_pooled)  # [B, 1, 480, 240]
        F_RL = F_RL.squeeze(1)  # [B, 480, 240] - spatial becomes features
        
        # Concatenate: [B, 480, 128+240=368]
        F_RG_L = torch.cat([F_RG, F_RL], dim=2)
        
        # Add positional embeddings
        F_RG_L = F_RG_L + self.h_pos_embed
        
        # VERTICAL FEATURES (for column splitting)
        # Global: learnable weighted average along height dimension  
        F_CG = torch.einsum('bchw,h->bcw', F_half, self.v_global_weight)  # [B, 128, 480]
        F_CG = F_CG.transpose(1, 2)  # [B, 480, 128]
        
        # Local: 2×1 AvgPool then 1×1 conv to 1 channel, treat spatial as features
        F_CL_pooled = F.avg_pool2d(F_half, kernel_size=(2, 1))  # [B, 128, 240, 480]
        F_CL = self.v_local_conv(F_CL_pooled)  # [B, 1, 240, 480]
        F_CL = F_CL.squeeze(1)  # [B, 240, 480]
        F_CL = F_CL.transpose(1, 2)  # [B, 480, 240] - transpose to get spatial as features
        
        # Concatenate: [B, 480, 128+240=368]
        F_CG_L = torch.cat([F_CG, F_CL], dim=2)

        # Add positional embeddings
        F_CG_L = F_CG_L + self.v_pos_embed
        
        # Transformer processing
        F_R = self.h_transformer(F_RG_L)  # [B, 480, 368]
        F_C = self.v_transformer(F_CG_L)  # [B, 480, 368]
        
        # Binary classification
        h_logits = self.h_classifier(F_R).squeeze(-1)  # [B, 480]
        v_logits = self.v_classifier(F_C).squeeze(-1)  # [B, 480]
        
        # 2× upsampling via duplication (as mentioned in paper)
        h_pred = h_logits.repeat_interleave(2, dim=1)  # [B, 960]
        v_pred = v_logits.repeat_interleave(2, dim=1)  # [B, 960]
        
        return torch.sigmoid(h_pred), torch.sigmoid(v_pred)

def focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """Focal loss exactly as specified in paper"""
    ce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
    pt = torch.where(targets == 1, predictions, 1 - predictions)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()

def post_process_with_ocr(h_pred, v_pred, ocr_text_centers, threshold=0.5):
    """
    OCR-based post-processing as mentioned in paper:
    If non-split regions contain no text projection, reclassify as split
    """
    h_binary = (h_pred > threshold).float()
    v_binary = (v_pred > threshold).float()
    
    # For each non-split region, check if it contains text
    # This is a simplified version - full implementation would need more detail
    
    return h_binary, v_binary

class TableDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

        self.transform = transforms.Compose([
            transforms.Resize((960, 960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Get PIL image and transform it
        image = item['image'].convert('RGB')
        image = self.transform(image)

        # Get HTML tags and cells data
        html_tags = item['html']
        cells = item['cells']

        # Generate ground truth
        h_gt, v_gt = get_ground_truth(item['image'], html_tags, cells)

        return image, torch.tensor(h_gt, dtype=torch.float), torch.tensor(v_gt, dtype=torch.float)

def setup_logging(log_dir):
    """Setup logging and tensorboard"""
    os.makedirs(log_dir, exist_ok=True)

    # Setup console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    return writer

def evaluate_model(model, val_dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_h_loss = 0
    total_v_loss = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, h_gt, v_gt in val_dataloader:
            images, h_gt, v_gt = images.to(device), h_gt.to(device), v_gt.to(device)

            h_pred, v_pred = model(images)

            h_loss = focal_loss(h_pred, h_gt, alpha=1.0, gamma=2.0)
            v_loss = focal_loss(v_pred, v_gt, alpha=1.0, gamma=2.0)
            batch_loss = h_loss + v_loss

            total_h_loss += h_loss.item()
            total_v_loss += v_loss.item()
            total_loss += batch_loss.item()
            num_batches += 1

    return {
        'total_loss': total_loss / num_batches,
        'h_loss': total_h_loss / num_batches,
        'v_loss': total_v_loss / num_batches
    }

def train_split_model(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Setup logging and tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/tablet_split_{timestamp}"
    writer = setup_logging(log_dir)

    # Initialize model
    model = SplitModel().to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Create train/val datasets
    train_dataset = TableDataset(ds['train'])
    val_dataset = TableDataset(ds['val'])

    # Subset for testing if specified
    if args.num_images > 0:
        logging.info(f"Using subset of {args.num_images} images for training")
        train_indices = list(range(min(args.num_images, len(train_dataset))))
        val_indices = list(range(min(args.num_images // 5, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Optimizer with exact parameters from paper
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-4
    )

    # Training tracking
    best_val_loss = float('inf')
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        epoch_h_loss = 0
        epoch_v_loss = 0
        epoch_total_loss = 0
        num_batches = 0

        for batch_idx, (images, h_gt, v_gt) in enumerate(train_dataloader):
            images, h_gt, v_gt = images.to(device), h_gt.to(device), v_gt.to(device)

            optimizer.zero_grad()

            h_pred, v_pred = model(images)

            # Calculate focal loss for both directions
            h_loss = focal_loss(h_pred, h_gt, alpha=1.0, gamma=2.0)
            v_loss = focal_loss(v_pred, v_gt, alpha=1.0, gamma=2.0)
            total_loss = h_loss + v_loss

            total_loss.backward()

            # Gradient clipping as mentioned in paper
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # Track losses
            epoch_h_loss += h_loss.item()
            epoch_v_loss += v_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            # Log to tensorboard
            if args.tensorboard:
                writer.add_scalar('Train/H_Loss_Step', h_loss.item(), global_step)
                writer.add_scalar('Train/V_Loss_Step', v_loss.item(), global_step)
                writer.add_scalar('Train/Total_Loss_Step', total_loss.item(), global_step)

            global_step += 1

            if batch_idx % args.log_interval == 0:
                log_msg = f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {total_loss:.4f}, H: {h_loss:.4f}, V: {v_loss:.4f}'
                logging.info(log_msg)
                print(log_msg)

        # Epoch averages
        avg_h_loss = epoch_h_loss / num_batches
        avg_v_loss = epoch_v_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        # Validation
        val_metrics = evaluate_model(model, val_dataloader, device)

        logging.info(f'Epoch {epoch+1}/{args.epochs} - '
                   f'Train Loss: {avg_total_loss:.4f} (H: {avg_h_loss:.4f}, V: {avg_v_loss:.4f}) | '
                   f'Val Loss: {val_metrics["total_loss"]:.4f} (H: {val_metrics["h_loss"]:.4f}, V: {val_metrics["v_loss"]:.4f})')

        # Log to tensorboard
        if args.tensorboard:
            writer.add_scalar('Train/H_Loss_Epoch', avg_h_loss, epoch)
            writer.add_scalar('Train/V_Loss_Epoch', avg_v_loss, epoch)
            writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
            writer.add_scalar('Val/H_Loss_Epoch', val_metrics['h_loss'], epoch)
            writer.add_scalar('Val/V_Loss_Epoch', val_metrics['v_loss'], epoch)
            writer.add_scalar('Val/Total_Loss_Epoch', val_metrics['total_loss'], epoch)

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            logging.info(f'New best model saved with val loss: {best_val_loss:.4f}')

        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
            }, checkpoint_path)
            logging.info(f'Checkpoint saved: {checkpoint_path}')

        model.train()  # Back to training mode

    writer.close()

    # Save final model
    final_model_path = os.path.join(log_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Final model saved: {final_model_path}')

    return model, log_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Train TABLET Split Detection Model')

    # Dataset arguments
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Number of images to train on (-1 for full dataset)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=16,
                        help='Number of epochs to train (default: 16)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')

    # Logging arguments
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval for batches (default: 10)')

    # Checkpointing arguments
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("🚀 Starting TABLET Split Detection Training")
    print(f"📊 Config: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    if args.num_images > 0:
        print(f"🔬 Testing mode: Training on {args.num_images} images only")
    else:
        print("🎯 Full training mode: Using complete dataset")

    if args.tensorboard:
        print("📈 Tensorboard logging enabled")

    # Train the model
    model, log_dir = train_split_model(args)

    print(f"✅ Training completed! Logs and models saved in: {log_dir}")
    print(f"📂 Best model: {log_dir}/best_model.pth")
    print(f"📂 Final model: {log_dir}/final_model.pth")

    if args.tensorboard:
        print(f"📈 View tensorboard: tensorboard --logdir {log_dir}")
