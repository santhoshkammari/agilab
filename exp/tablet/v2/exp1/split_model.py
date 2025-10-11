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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def get_ground_truth(image, cells, otsl, split_width=5):

    """
    parse OTSL to derive row/column split positions.
    this is the groundtruth for split model training.

    Args:
        image: PIL Image
        html_tags: not used, kept for compatibility
        cells: nested list - cells[0] contains actual cell data
        otsl: OTSL token sequence
        split_width: width of split regions in pixels (default: 5)
    """
    orig_width, orig_height = image.size
    target_size = 960
    
    # cells is nested - extract actual list
    cells_flat = cells[0]
    
    # parse OTSL to build 2D grid
    grid = []
    current_row = []
    cell_idx = 0  # only increments for fcel ,ecel tokens
    
    for token in otsl:
        if token == 'nl':
            if current_row:
                grid.append(current_row)
                current_row = []
        elif token == 'fcel' or token=='ecel':
            current_row.append({'type': token, 'cell_idx': cell_idx})
            cell_idx += 1
        elif token in ['lcel', 'ucel', 'xcel']:
            # merge/empty tokens don't consume bboxes
            current_row.append({'type': token, 'cell_idx': None})
    
    if current_row:
        grid.append(current_row)
    
    # derive row splits - max y2 for each row
    row_splits = []
    for row in grid:
        row_cell_indices = [item['cell_idx'] for item in row if item['cell_idx'] is not None]
        if row_cell_indices:
            max_y = max(cells_flat[i]['bbox'][3] for i in row_cell_indices)
            row_splits.append(max_y)
    
    # derive column splits - max x2 for each column
    num_cols = len(grid[0]) if grid else 0
    col_splits = []
    for col_idx in range(num_cols):
        col_max_x = []
        for row in grid:
            if col_idx < len(row) and row[col_idx]['cell_idx'] is not None:
                next_is_lcel = (col_idx + 1 < len(row) and row[col_idx + 1]['type'] == 'lcel')
                if not next_is_lcel:
                    cell_id = row[col_idx]['cell_idx']
                    col_max_x.append(cells_flat[cell_id]['bbox'][2])
        if col_max_x:
            col_splits.append(max(col_max_x))

    # # DEBUG: print what we found
    # print(f"\nDEBUG get_ground_truth:")
    # print(f"  Found {len(row_splits)} row splits: {row_splits}")
    # print(f"  Found {len(col_splits)} col splits: {col_splits}")
    
    # # scale to target size
    # y_scaled = [(y * target_size / orig_height) for y in row_splits]
    # x_scaled = [(x * target_size / orig_width) for x in col_splits]
    
    # print(f"  Scaled row splits: {[int(y) for y in y_scaled]}")
    # print(f"  Scaled col splits: {[int(x) for x in x_scaled]}")

    
    row_splits = row_splits[:-1]
    col_splits = col_splits[:-1]

    # scale to target size
    y_scaled = [(y * target_size / orig_height) for y in row_splits]
    x_scaled = [(x * target_size / orig_width) for x in col_splits]
    
    # init ground truth arrays
    horizontal_gt = [0] * target_size
    vertical_gt = [0] * target_size

    all_x1 = [c['bbox'][0] for c in cells_flat]
    all_y1 = [c['bbox'][1] for c in cells_flat]
    all_x2 = [c['bbox'][2] for c in cells_flat]
    all_y2 = [c['bbox'][3] for c in cells_flat]
    table_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]
    table_y1 = int(round(table_bbox[1] * target_size / orig_height))
    table_y2 = int(round(table_bbox[3] * target_size / orig_height))
    table_x1 = int(round(table_bbox[0] * target_size / orig_width))
    table_x2 = int(round(table_bbox[2] * target_size / orig_width))


    # Mark table bbox boundaries (5 pixels wide)
    # Top boundary
    for offset in range(split_width):
        pos = table_y1 + offset
        if 0 <= pos < target_size:
            horizontal_gt[pos] = 1

    # Bottom boundary
    for offset in range(split_width):
        pos = table_y2 - offset
        if 0 <= pos < target_size:
            horizontal_gt[pos] = 1

    # Left boundary
    for offset in range(split_width):
        pos = table_x1 + offset
        if 0 <= pos < target_size:
            vertical_gt[pos] = 1

    # Right boundary
    for offset in range(split_width):
        pos = table_x2 - offset
        if 0 <= pos < target_size:
            vertical_gt[pos] = 1

    # mark split regions (configurable pixel width)
    for y in y_scaled:
        y_int = int(round(y))
        if 0 <= y_int < target_size:
            for offset in range(split_width):
                pos = y_int + offset
                if 0 <= pos < target_size:
                    horizontal_gt[pos] = 1

    for x in x_scaled:
        x_int = int(round(x))
        if 0 <= x_int < target_size:
            for offset in range(split_width):
                pos = x_int + offset
                if 0 <= pos < target_size:
                    vertical_gt[pos] = 1
    
    return horizontal_gt, vertical_gt


def get_ground_truth_auto_gap(image, cells, otsl):
    """
    Parse OTSL to derive row/column split positions with DYNAMIC gap widths.
    This creates ground truth for the split model training.

    Args:
        image: PIL Image
        cells: nested list - cells[0] contains actual cell data
        otsl: OTSL token sequence
    """
    orig_width, orig_height = image.size
    target_size = 960
    
    # cells is nested - extract actual list
    cells_flat = cells[0]
    
    # Parse OTSL to build 2D grid
    grid = []
    current_row = []
    cell_idx = 0  # only increments for fcel, ecel tokens
    
    for token in otsl:
        if token == 'nl':
            if current_row:
                grid.append(current_row)
                current_row = []
        elif token == 'fcel' or token == 'ecel':
            current_row.append({'type': token, 'cell_idx': cell_idx})
            cell_idx += 1
        elif token in ['lcel', 'ucel', 'xcel']:
            # merge/empty tokens don't consume bboxes
            current_row.append({'type': token, 'cell_idx': None})
    
    if current_row:
        grid.append(current_row)
    
    # Get row boundaries (min y1 and max y2 for each row)
    row_boundaries = []
    for row in grid:
        row_cell_indices = [item['cell_idx'] for item in row if item['cell_idx'] is not None]
        if row_cell_indices:
            min_y1 = min(cells_flat[i]['bbox'][1] for i in row_cell_indices)
            max_y2 = max(cells_flat[i]['bbox'][3] for i in row_cell_indices)
            row_boundaries.append({'min_y': min_y1, 'max_y': max_y2})
    
    # Get column boundaries (min x1 and max x2 for each column)
    num_cols = len(grid[0]) if grid else 0
    col_boundaries = []
    for col_idx in range(num_cols):
        col_cells = []
        for row in grid:
            if col_idx < len(row) and row[col_idx]['cell_idx'] is not None:
                # Check if next cell is lcel (merged left)
                next_is_lcel = (col_idx + 1 < len(row) and row[col_idx + 1]['type'] == 'lcel')
                if not next_is_lcel:
                    cell_id = row[col_idx]['cell_idx']
                    col_cells.append(cell_id)
        if col_cells:
            min_x1 = min(cells_flat[i]['bbox'][0] for i in col_cells)
            max_x2 = max(cells_flat[i]['bbox'][2] for i in col_cells)
            col_boundaries.append({'min_x': min_x1, 'max_x': max_x2})
    
    # Calculate table bbox
    all_x1 = [c['bbox'][0] for c in cells_flat]
    all_y1 = [c['bbox'][1] for c in cells_flat]
    all_x2 = [c['bbox'][2] for c in cells_flat]
    all_y2 = [c['bbox'][3] for c in cells_flat]
    table_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]
    
    # Init ground truth arrays
    horizontal_gt = [0] * target_size
    vertical_gt = [0] * target_size
    
    # Helper function to scale and mark range
    def mark_range(gt_array, start, end, orig_dim):
        """Mark all pixels from start to end (scaled to target_size)"""
        start_scaled = int(round(start * target_size / orig_dim))
        end_scaled = int(round(end * target_size / orig_dim))
        for pos in range(start_scaled, min(end_scaled + 1, target_size)):
            if 0 <= pos < target_size:
                gt_array[pos] = 1
    
    # Mark HORIZONTAL gaps (between rows)
    # 1. Gap from image top to first row top
    if row_boundaries:
        mark_range(horizontal_gt, 0, row_boundaries[0]['min_y'], orig_height)
    
    # 2. Gaps between consecutive rows
    for i in range(len(row_boundaries) - 1):
        gap_start = row_boundaries[i]['max_y']
        gap_end = row_boundaries[i + 1]['min_y']
        if gap_end > gap_start:  # Only mark if there's actual gap
            mark_range(horizontal_gt, gap_start, gap_end, orig_height)
    
    # 3. Gap from last row bottom to image bottom
    if row_boundaries:
        mark_range(horizontal_gt, row_boundaries[-1]['max_y'], orig_height, orig_height)
    
    # Mark VERTICAL gaps (between columns)
    # 1. Gap from image left to first column left
    if col_boundaries:
        mark_range(vertical_gt, 0, col_boundaries[0]['min_x'], orig_width)
    
    # 2. Gaps between consecutive columns
    for i in range(len(col_boundaries) - 1):
        gap_start = col_boundaries[i]['max_x']
        gap_end = col_boundaries[i + 1]['min_x']
        if gap_end > gap_start:  # Only mark if there's actual gap
            mark_range(vertical_gt, gap_start, gap_end, orig_width)
    
    # 3. Gap from last column right to image right
    if col_boundaries:
        mark_range(vertical_gt, col_boundaries[-1]['max_x'], orig_width, orig_width)
    
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
        
        # Fix: Correct feature dimensions - 128 + W/4 = 128 + 120 = 248
        feature_dim = 128 + 120  # Global + Local features

        # Positional embeddings (1D as mentioned in paper)
        self.h_pos_embed = nn.Parameter(torch.randn(480, feature_dim))
        self.v_pos_embed = nn.Parameter(torch.randn(480, feature_dim))

        # Transformers with correct dimensions
        self.h_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )
        self.v_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )

        # Classification heads
        self.h_classifier = nn.Linear(feature_dim, 1)
        self.v_classifier = nn.Linear(feature_dim, 1)
        
    def forward(self, x):
        # Input: [B, 3, 960, 960]
        features = self.backbone(x)      # [B, 256, 60, 60]
        F_half = self.fpn(features)      # [B, 128, 480, 480] - This is F1/2
        
        B, C, H, W = F_half.shape        # B, 128, 480, 480
        
        # HORIZONTAL FEATURES (for row splitting)
        # Global: learnable weighted average along width dimension
        F_RG = torch.einsum('bchw,w->bch', F_half, self.h_global_weight)  # [B, 128, 480]
        F_RG = F_RG.transpose(1, 2)  # [B, 480, 128]
        
        # Local: 1×4 AvgPool to get 120 features (W/4), then 1×1 conv to 1 channel
        F_RL_pooled = F.avg_pool2d(F_half, kernel_size=(1, 4))  # [B, 128, 480, 120]
        F_RL = self.h_local_conv(F_RL_pooled)  # [B, 1, 480, 120]
        F_RL = F_RL.squeeze(1)  # [B, 480, 120] - spatial becomes features

        # Concatenate: [B, 480, 128+120=248]
        F_RG_L = torch.cat([F_RG, F_RL], dim=2)
        
        # Add positional embeddings
        F_RG_L = F_RG_L + self.h_pos_embed
        
        # VERTICAL FEATURES (for column splitting)
        # Global: learnable weighted average along height dimension  
        F_CG = torch.einsum('bchw,h->bcw', F_half, self.v_global_weight)  # [B, 128, 480]
        F_CG = F_CG.transpose(1, 2)  # [B, 480, 128]
        
        # Local: 4×1 AvgPool to get 120 features (H/4), then 1×1 conv to 1 channel
        F_CL_pooled = F.avg_pool2d(F_half, kernel_size=(4, 1))  # [B, 128, 120, 480]
        F_CL = self.v_local_conv(F_CL_pooled)  # [B, 1, 120, 480]
        F_CL = F_CL.squeeze(1)  # [B, 120, 480]
        F_CL = F_CL.transpose(1, 2)  # [B, 480, 120] - transpose to get spatial as features

        # Concatenate: [B, 480, 128+120=248]
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

def post_process_predictions(h_pred, v_pred, threshold=0.5):
    """
    Simple post-processing to convert predictions to binary masks
    """
    h_binary = (h_pred > threshold).float()
    v_binary = (v_pred > threshold).float()

    return h_binary, v_binary

class TableDataset(Dataset):
    def __init__(self, hf_dataset, split_width=5):
        self.hf_dataset = hf_dataset
        self.split_width = split_width

        self.transform = transforms.Compose([
            transforms.Resize((960, 960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        image = item['image'].convert('RGB')
        image_transformed = self.transform(image)

        # pass otsl to ground truth generation with configurable split_width
        h_gt, v_gt = get_ground_truth_auto_gap(
            item['image'],  # original PIL image for dimensions
            item['cells'],
            item['otsl'],
            # split_width=self.split_width
        )

        return image_transformed, torch.tensor(h_gt, dtype=torch.float), torch.tensor(v_gt, dtype=torch.float)

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

def visualize_predictions(model, dataset, device, idx=0, threshold=0.5):
    """
    run inference on one image and draw predicted split lines.
    
    Args:
        model: trained SplitModel
        dataset: TableDataset instance
        device: cuda/cpu
        idx: which image to visualize
        threshold: binary threshold for predictions
    """
    model.eval()
    
    # get one sample
    image_tensor, h_gt, v_gt = dataset[idx]
    
    # get original image for drawing
    original_item = dataset.hf_dataset[idx]
    img = original_item['image'].convert('RGB')
    
    # resize to 960x960 for visualization
    img_resized = img.resize((960, 960))
    draw = ImageDraw.Draw(img_resized)
    
    # run inference
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, 960, 960]
        h_pred, v_pred = model(image_batch)
        h_pred = h_pred.squeeze(0).cpu().numpy()  # [960]
        v_pred = v_pred.squeeze(0).cpu().numpy()  # [960]
    
    # convert predictions to binary
    h_binary = (h_pred > threshold).astype(int)
    v_binary = (v_pred > threshold).astype(int)
    
    # draw ground truth in green (thin)
    h_gt_np = h_gt.numpy()
    v_gt_np = v_gt.numpy()
    
    # for y in range(960):
    #     if h_gt_np[y] == 1:
    #         draw.line([(0, y), (960, y)], fill='yellow', width=2)
    
    # for x in range(960):
    #     if v_gt_np[x] == 1:
    #         draw.line([(x, 0), (x, 960)], fill='yellow', width=2)
    
    # draw predictions in red (thicker so you can see both)
    for y in range(960):
        if h_binary[y] == 1:
            draw.line([(0, y), (960, y)], fill='yellow', width=2)
    
    for x in range(960):
        if v_binary[x] == 1:
            draw.line([(x, 0), (x, 960)], fill='yellow', width=2)
    
    # show stats
    h_matches = (h_binary == h_gt_np).sum()
    v_matches = (v_binary == v_gt_np).sum()
    
    print(f"\nvisualization stats for image {idx}:")
    print(f"  horizontal accuracy: {h_matches/960:.3f} ({h_matches}/960 pixels)")
    print(f"  vertical accuracy: {v_matches/960:.3f} ({v_matches}/960 pixels)")
    print(f"  ground truth split lines: H={h_gt_np.sum()}, V={v_gt_np.sum()}")
    print(f"  predicted split lines: H={h_binary.sum()}, V={v_binary.sum()}")
    print(f"\ngreen = ground truth, red = predictions")
    
    plt.figure(figsize=(15, 15))
    plt.imshow(img_resized)
    plt.title(f'Split Predictions - Image {idx}')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'split_predictions_{idx}.png', bbox_inches='tight')
    # plt.show()  
    
    return img_resized

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

    # Load dataset with proper train/val split
    from datasets import load_dataset
    logger.info("📥 Loading FinTabNet dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Fix: Use proper dataset splits
    train_dataset = TableDataset(ds['train'])
    val_dataset = TableDataset(ds['val'])  # Dataset already has val split

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


    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    return parser.parse_args()


