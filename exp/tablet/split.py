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
    Generate ground truth for split lines following TABLET methodology

    Args:
        image: PIL Image object
        html_tags: list of HTML tags from dataset (should use html_restored)
        cells: list of cell data with tokens and bboxes from dataset

    Returns:
        horizontal_gt: list of 960 binary values (0/1) for row splits
        vertical_gt: list of 960 binary values (0/1) for column splits
    """
    # Get original image dimensions
    orig_width, orig_height = image.size

    # Parse HTML structure
    html_content = ''.join(html_tags)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text block centers from cells (cells has nested structure)
    text_centers = []
    # Flatten cells structure - cells[0] contains the actual list of cells
    actual_cells = cells[0] if len(cells) > 0 and isinstance(cells[0], list) else cells
    for cell in actual_cells:
        tokens = cell['tokens']
        bbox = cell['bbox']
        text = ''.join(tokens)

        # Skip empty cells
        if not text.strip():
            continue

        # bbox format: [x1, y1, x2, y2, page]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Calculate text center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Scale to 960x960 coordinates
        scaled_x = int(center_x * 960 / orig_width)
        scaled_y = int(center_y * 960 / orig_height)

        text_centers.append((scaled_x, scaled_y, text))

    # Initialize ground truth arrays
    horizontal_gt = [0] * 960
    vertical_gt = [0] * 960

    # TABLET methodology: Find split regions based on text projections

    # HORIZONTAL SPLITS (row boundaries)
    # Project text centers onto Y-axis
    y_projections = [center[1] for center in text_centers]
    y_projections.sort()

    # Find gaps between consecutive Y projections
    if len(y_projections) > 1:
        for i in range(len(y_projections) - 1):
            curr_y = y_projections[i]
            next_y = y_projections[i + 1]

            # If gap is significant (> 10 pixels), mark as split region
            if next_y - curr_y > 10:
                split_start = curr_y + 5  # Small offset from text
                split_end = next_y - 5

                # Ensure minimum split width of 5 pixels
                if split_end - split_start < 5:
                    split_center = (split_start + split_end) // 2
                    split_start = split_center - 2
                    split_end = split_center + 3

                # Mark split region
                for y in range(max(0, split_start), min(960, split_end)):
                    horizontal_gt[y] = 1

    # VERTICAL SPLITS (column boundaries)
    # Project text centers onto X-axis
    x_projections = [center[0] for center in text_centers]
    x_projections.sort()

    # Find gaps between consecutive X projections
    if len(x_projections) > 1:
        for i in range(len(x_projections) - 1):
            curr_x = x_projections[i]
            next_x = x_projections[i + 1]

            # If gap is significant (> 10 pixels), mark as split region
            if next_x - curr_x > 10:
                split_start = curr_x + 5  # Small offset from text
                split_end = next_x - 5

                # Ensure minimum split width of 5 pixels
                if split_end - split_start < 5:
                    split_center = (split_start + split_end) // 2
                    split_start = split_center - 2
                    split_end = split_center + 3

                # Mark split region
                for x in range(max(0, split_start), min(960, split_end)):
                    vertical_gt[x] = 1

    # TABLET post-processing: If regions have no text projections, mark as splits
    # This step ensures robustness for tables with missing content

    # Check for empty horizontal regions
    for y in range(0, 960, 10):  # Sample every 10 pixels
        has_text_nearby = any(abs(center[1] - y) < 15 for center in text_centers)
        if not has_text_nearby and y > 50 and y < 910:  # Avoid image borders
            # Mark small split region around this empty area
            for split_y in range(max(0, y-2), min(960, y+3)):
                horizontal_gt[split_y] = 1

    # Check for empty vertical regions
    for x in range(0, 960, 10):  # Sample every 10 pixels
        has_text_nearby = any(abs(center[0] - x) < 15 for center in text_centers)
        if not has_text_nearby and x > 50 and x < 910:  # Avoid image borders
            # Mark small split region around this empty area
            for split_x in range(max(0, x-2), min(960, x+3)):
                vertical_gt[split_x] = 1

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
        html_tags = item['html_restored']
        cells = item['cells']

        # Generate ground truth
        h_gt, v_gt = get_ground_truth(item['image'], html_tags, cells)

        return image, torch.tensor(h_gt, dtype=torch.float), torch.tensor(v_gt, dtype=torch.float)

def setup_logging(log_dir):
    """Setup comprehensive logging and tensorboard"""
    os.makedirs(log_dir, exist_ok=True)

    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create custom logger
    logger = logging.getLogger('TABLET')
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
    writer = SummaryWriter(log_dir, comment='_TABLET_Split_Detection')

    # Log system information
    logger.info("=" * 60)
    logger.info("🚀 TABLET Split Detection Training Started")
    logger.info("=" * 60)
    logger.info(f"📁 Log directory: {log_dir}")
    logger.info(f"🔧 PyTorch version: {torch.__version__}")
    logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU device: {torch.cuda.get_device_name()}")
        logger.info(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return writer, logger

def evaluate_model(model, val_dataloader, device, logger=None):
    """Evaluate model on validation set with comprehensive metrics"""
    model.eval()
    total_h_loss = 0
    total_v_loss = 0
    total_loss = 0

    # Additional metrics
    h_accuracy_sum = 0
    v_accuracy_sum = 0
    h_precision_sum = 0
    v_precision_sum = 0
    h_recall_sum = 0
    v_recall_sum = 0

    num_batches = 0

    if logger:
        logger.info("🔍 Starting validation evaluation...")

    with torch.no_grad():
        for batch_idx, (images, h_gt, v_gt) in enumerate(val_dataloader):
            images, h_gt, v_gt = images.to(device), h_gt.to(device), v_gt.to(device)

            h_pred, v_pred = model(images)

            # Calculate losses
            h_loss = focal_loss(h_pred, h_gt, alpha=1.0, gamma=2.0)
            v_loss = focal_loss(v_pred, v_gt, alpha=1.0, gamma=2.0)
            batch_loss = h_loss + v_loss

            total_h_loss += h_loss.item()
            total_v_loss += v_loss.item()
            total_loss += batch_loss.item()

            # Calculate additional metrics
            h_pred_binary = (h_pred > 0.5).float()
            v_pred_binary = (v_pred > 0.5).float()

            # Accuracy
            h_accuracy = (h_pred_binary == h_gt).float().mean()
            v_accuracy = (v_pred_binary == v_gt).float().mean()
            h_accuracy_sum += h_accuracy.item()
            v_accuracy_sum += v_accuracy.item()

            # Precision and Recall (for split regions)
            h_tp = ((h_pred_binary == 1) & (h_gt == 1)).sum().float()
            h_fp = ((h_pred_binary == 1) & (h_gt == 0)).sum().float()
            h_fn = ((h_pred_binary == 0) & (h_gt == 1)).sum().float()

            v_tp = ((v_pred_binary == 1) & (v_gt == 1)).sum().float()
            v_fp = ((v_pred_binary == 1) & (v_gt == 0)).sum().float()
            v_fn = ((v_pred_binary == 0) & (v_gt == 1)).sum().float()

            # Avoid division by zero
            h_precision = h_tp / (h_tp + h_fp + 1e-8)
            h_recall = h_tp / (h_tp + h_fn + 1e-8)
            v_precision = v_tp / (v_tp + v_fp + 1e-8)
            v_recall = v_tp / (v_tp + v_fn + 1e-8)

            h_precision_sum += h_precision.item()
            h_recall_sum += h_recall.item()
            v_precision_sum += v_precision.item()
            v_recall_sum += v_recall.item()

            num_batches += 1

            # Log progress every 20 batches
            if logger and batch_idx % 20 == 0:
                logger.debug(f"Validation batch {batch_idx}/{len(val_dataloader)}")

    # Calculate averages
    metrics = {
        'total_loss': total_loss / num_batches,
        'h_loss': total_h_loss / num_batches,
        'v_loss': total_v_loss / num_batches,
        'h_accuracy': h_accuracy_sum / num_batches,
        'v_accuracy': v_accuracy_sum / num_batches,
        'h_precision': h_precision_sum / num_batches,
        'v_precision': v_precision_sum / num_batches,
        'h_recall': h_recall_sum / num_batches,
        'v_recall': v_recall_sum / num_batches,
    }

    # Calculate F1 scores
    metrics['h_f1'] = 2 * (metrics['h_precision'] * metrics['h_recall']) / (metrics['h_precision'] + metrics['h_recall'] + 1e-8)
    metrics['v_f1'] = 2 * (metrics['v_precision'] * metrics['v_recall']) / (metrics['v_precision'] + metrics['v_recall'] + 1e-8)

    if logger:
        logger.info("✅ Validation completed")

    return metrics

def train_split_model(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging and tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/tablet_split_{timestamp}"
    writer, logger = setup_logging(log_dir)

    logger.info(f"🎮 Using device: {device}")

    # Initialize model
    model = SplitModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

    # Load dataset
    from datasets import load_dataset
    logger.info("📥 Loading FinTabNet dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Create train/val datasets
    train_dataset = TableDataset(ds['train'])
    val_dataset = TableDataset(ds['val'])

    # Subset for testing if specified
    if args.num_images > 0:
        logger.info(f"🔬 Using subset of {args.num_images} images for training")
        train_indices = list(range(min(args.num_images, len(train_dataset))))
        val_indices = list(range(min(args.num_images // 5, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    logger.info(f"📊 Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # Log dataset info to tensorboard
    writer.add_scalar('Dataset/Train_Size', len(train_dataset), 0)
    writer.add_scalar('Dataset/Val_Size', len(val_dataset), 0)
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Optimizer with exact parameters from paper
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
    best_val_f1 = 0.0
    global_step = 0
    start_time = datetime.now()

    logger.info("🚀 Starting training loop...")
    model.train()
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        epoch_h_loss = 0
        epoch_v_loss = 0
        epoch_total_loss = 0
        num_batches = 0

        logger.info(f"📖 Epoch {epoch+1}/{args.epochs} started")

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

            # Enhanced tensorboard logging
            writer.add_scalar('Train/H_Loss_Step', h_loss.item(), global_step)
            writer.add_scalar('Train/V_Loss_Step', v_loss.item(), global_step)
            writer.add_scalar('Train/Total_Loss_Step', total_loss.item(), global_step)
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

            # Log every batch
            elapsed = datetime.now() - epoch_start_time
            progress = (batch_idx + 1) / len(train_dataloader) * 100
            log_msg = f'Epoch {epoch+1}/{args.epochs} [{progress:5.1f}%] Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {total_loss:.4f} (H: {h_loss:.4f}, V: {v_loss:.4f}) | Elapsed: {elapsed}'
            logger.info(log_msg)

        # Epoch averages
        avg_h_loss = epoch_h_loss / num_batches
        avg_v_loss = epoch_v_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        # Validation
        epoch_duration = datetime.now() - epoch_start_time
        logger.info(f"🗓️ Epoch {epoch+1} training completed in {epoch_duration}")
        val_metrics = evaluate_model(model, val_dataloader, device, logger)

        # Comprehensive epoch logging
        logger.info(f"📊 Epoch {epoch+1}/{args.epochs} Results:")
        logger.info(f"   Train Loss: {avg_total_loss:.4f} (H: {avg_h_loss:.4f}, V: {avg_v_loss:.4f})")
        logger.info(f"   Val Loss: {val_metrics['total_loss']:.4f} (H: {val_metrics['h_loss']:.4f}, V: {val_metrics['v_loss']:.4f})")
        logger.info(f"   Val Accuracy: H: {val_metrics['h_accuracy']:.3f}, V: {val_metrics['v_accuracy']:.3f}")
        logger.info(f"   Val F1 Score: H: {val_metrics['h_f1']:.3f}, V: {val_metrics['v_f1']:.3f}")
        logger.info(f"   Val Precision: H: {val_metrics['h_precision']:.3f}, V: {val_metrics['v_precision']:.3f}")
        logger.info(f"   Val Recall: H: {val_metrics['h_recall']:.3f}, V: {val_metrics['v_recall']:.3f}")

        # Enhanced tensorboard logging
        # Loss metrics
        writer.add_scalar('Train/H_Loss_Epoch', avg_h_loss, epoch)
        writer.add_scalar('Train/V_Loss_Epoch', avg_v_loss, epoch)
        writer.add_scalar('Train/Total_Loss_Epoch', avg_total_loss, epoch)
        writer.add_scalar('Val/H_Loss_Epoch', val_metrics['h_loss'], epoch)
        writer.add_scalar('Val/V_Loss_Epoch', val_metrics['v_loss'], epoch)
        writer.add_scalar('Val/Total_Loss_Epoch', val_metrics['total_loss'], epoch)

        # Performance metrics
        writer.add_scalar('Val/H_Accuracy', val_metrics['h_accuracy'], epoch)
        writer.add_scalar('Val/V_Accuracy', val_metrics['v_accuracy'], epoch)
        writer.add_scalar('Val/H_F1_Score', val_metrics['h_f1'], epoch)
        writer.add_scalar('Val/V_F1_Score', val_metrics['v_f1'], epoch)
        writer.add_scalar('Val/H_Precision', val_metrics['h_precision'], epoch)
        writer.add_scalar('Val/V_Precision', val_metrics['v_precision'], epoch)
        writer.add_scalar('Val/H_Recall', val_metrics['h_recall'], epoch)
        writer.add_scalar('Val/V_Recall', val_metrics['v_recall'], epoch)

        # Combined metrics
        avg_f1 = (val_metrics['h_f1'] + val_metrics['v_f1']) / 2
        avg_accuracy = (val_metrics['h_accuracy'] + val_metrics['v_accuracy']) / 2
        writer.add_scalar('Val/Average_F1_Score', avg_f1, epoch)
        writer.add_scalar('Val/Average_Accuracy', avg_accuracy, epoch)

        # Save best model (based on F1 score)
        avg_f1 = (val_metrics['h_f1'] + val_metrics['v_f1']) / 2
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_val_loss = val_metrics['total_loss']
            best_model_path = os.path.join(log_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            logger.info(f'✨ New best model saved! F1: {best_val_f1:.4f}, Loss: {best_val_loss:.4f}')

        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
            }, checkpoint_path)
            logger.info(f'💾 Checkpoint saved: {checkpoint_path}')

        model.train()  # Back to training mode

    writer.close()

    # Training completion
    total_training_time = datetime.now() - start_time
    logger.info("="*60)
    logger.info("🏆 Training Completed Successfully!")
    logger.info(f"⏱️ Total training time: {total_training_time}")
    logger.info(f"🎆 Best validation F1 score: {best_val_f1:.4f}")
    logger.info(f"🎆 Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)

    # Save final model
    final_model_path = os.path.join(log_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'💾 Final model saved: {final_model_path}')

    # Close tensorboard writer
    writer.close()
    logger.info("📈 Tensorboard logs closed")

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

    print("📈 Tensorboard logging enabled")

    # Train the model
    model, log_dir = train_split_model(args)

    print(f"✅ Training completed! Logs and models saved in: {log_dir}")
    print(f"📂 Best model: {log_dir}/best_model.pth")
    print(f"📂 Final model: {log_dir}/final_model.pth")

    print(f"📈 View tensorboard: tensorboard --logdir {log_dir}")
