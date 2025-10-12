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


def get_ground_truth_auto_gap_expand_min5pix_overlap_cells(image, cells, otsl, split_width=5):
    """
    Parse OTSL to derive row/column split positions with DYNAMIC gap widths.
    This creates ground truth for the split model training.

    Args:
        image: PIL Image
        cells: nested list - cells[0] contains actual cell data
        otsl: OTSL token sequence
        split_width: width of split when there's no gap (default: 5)
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
        elif token in ['fcel', 'ecel']:  # FIXED: was == ['fcel','ecel']
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
            row_boundaries.append({'min_y': min_y1, 'max_y': max_y2, 'row_cells': row_cell_indices})
    
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
            col_boundaries.append({'min_x': min_x1, 'max_x': max_x2, 'col_cells': col_cells})
    
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
        else:
            # No gap or overlap - find actual split position
            curr_row_y2 = [cells_flat[cell_id]['bbox'][3] for cell_id in row_boundaries[i]['row_cells']]
            next_row_y1 = [cells_flat[cell_id]['bbox'][1] for cell_id in row_boundaries[i + 1]['row_cells']]
            
            max_curr_y2 = max(curr_row_y2)
            min_next_y1 = min(next_row_y1)
            
            # Mark between the actual closest cells
            if min_next_y1 > max_curr_y2:
                mark_range(horizontal_gt, max_curr_y2, min_next_y1, orig_height)
            else:
                # Overlap - mark fixed width at midpoint
                split_pos = (max_curr_y2 + min_next_y1) / 2
                mark_range(horizontal_gt, split_pos - split_width/2, split_pos + split_width/2, orig_height)
    
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
        
        if gap_end > gap_start:  # Actual gap exists
            mark_range(vertical_gt, gap_start, gap_end, orig_width)
        else:
            # No gap or overlap - use col_cells to find actual split position
            curr_col_x2 = [cells_flat[cell_id]['bbox'][2] for cell_id in col_boundaries[i]['col_cells']]
            next_col_x1 = [cells_flat[cell_id]['bbox'][0] for cell_id in col_boundaries[i + 1]['col_cells']]
            
            max_curr_x2 = max(curr_col_x2)
            min_next_x1 = min(next_col_x1)
            
            # Mark between the actual closest cells
            if min_next_x1 > max_curr_x2:
                mark_range(vertical_gt, max_curr_x2, min_next_x1, orig_width)
            else:
                # Overlap case - mark fixed width at midpoint
                split_pos = (max_curr_x2 + min_next_x1) / 2
                mark_range(vertical_gt, split_pos - split_width/2, split_pos + split_width/2, orig_width)
    
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

        # Binary classification at 480 resolution
        h_logits = self.h_classifier(F_R).squeeze(-1)  # [B, 480]
        v_logits = self.v_classifier(F_C).squeeze(-1)  # [B, 480]

        # return at 480 resolution (upsample happens AFTER loss computation)
        return torch.sigmoid(h_logits), torch.sigmoid(v_logits)  # [B, 480]

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

        image = item['image'].convert('RGB')
        image_transformed = self.transform(image)

        # generate GT at 960 resolution
        h_gt_960, v_gt_960 = get_ground_truth_auto_gap(
            item['image'],  # original PIL image for dimensions
            item['cells'],
            item['otsl'],
        )

        # downsample to 480 for loss computation (take every 2nd element)
        h_gt_480 = [h_gt_960[i] for i in range(0, 960, 2)]  # [480]
        v_gt_480 = [v_gt_960[i] for i in range(0, 960, 2)]  # [480]

        return (
            image_transformed,
            torch.tensor(h_gt_480, dtype=torch.float),  # [480] for training loss
            torch.tensor(v_gt_480, dtype=torch.float),  # [480] for training loss
            torch.tensor(h_gt_960, dtype=torch.float),  # [960] for metrics
            torch.tensor(v_gt_960, dtype=torch.float),  # [960] for metrics
        )


