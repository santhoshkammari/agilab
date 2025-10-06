import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.ops import roi_align
from PIL import Image
import os
import numpy as np
import argparse
import logging
from datetime import datetime


def get_merge_ground_truth(otsl, R, C):
    """
    Convert OTSL sequence to grid cell labels for merge model training.

    Args:
        otsl: OTSL token sequence (e.g., ['fcel', 'lcel', 'nl', 'ucel', ...])
        R: number of rows in grid
        C: number of columns in grid

    Returns:
        grid_labels: tensor of shape (R, C) with class indices:
            0 = 'C' (fcel/ecel - new cell)
            1 = 'L' (lcel - merge left)
            2 = 'U' (ucel - merge up)
            3 = 'X' (xcel - merge both)
    """
    # Parse OTSL to build 2D grid
    grid = []
    current_row = []

    for token in otsl:
        if token == 'nl':
            if current_row:
                grid.append(current_row)
                current_row = []
        elif token == 'fcel':
            current_row.append('C')  # New cell
        elif token == 'lcel':
            current_row.append('L')  # Merge left
        elif token == 'ucel':
            current_row.append('U')  # Merge up
        elif token == 'xcel':
            current_row.append('X')  # Merge both
        elif token == 'ecel':
            current_row.append('C')  # Empty cell treated as new cell

    if current_row:
        grid.append(current_row)

    # Convert to class indices
    token_to_class = {'C': 0, 'L': 1, 'U': 2, 'X': 3}

    # Initialize grid labels
    grid_labels = torch.zeros(R, C, dtype=torch.long)

    # Fill in labels
    for i in range(min(R, len(grid))):
        for j in range(min(C, len(grid[i]))):
            grid_labels[i, j] = token_to_class[grid[i][j]]

    return grid_labels


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock"""
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


class ResNet18(nn.Module):
    """Standard ResNet-18 (not modified like Split model)"""
    def __init__(self):
        super().__init__()
        # First conv block - standard channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Keep maxpool

        # Standard ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # [B, 64, 480, 480]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 240, 240]

        x = self.layer1(x)   # [B, 64, 240, 240]
        x = self.layer2(x)   # [B, 128, 120, 120]
        x = self.layer3(x)   # [B, 256, 60, 60]
        x = self.layer4(x)   # [B, 512, 30, 30]
        return x


class FPN(nn.Module):
    """Feature Pyramid Network outputting 256 channels at H/4×W/4"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x):
        # x is [B, 512, 30, 30] from ResNet
        x = self.conv(x)  # [B, 256, 30, 30]
        # Upsample to H/4×W/4 = 240×240
        x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=False)
        return x  # [B, 256, 240, 240]


class MergeModel(nn.Module):
    def __init__(self, max_grid_cells=640):
        super().__init__()
        self.backbone = ResNet18()
        self.fpn = FPN()
        self.max_grid_cells = max_grid_cells

        # RoIAlign output size
        self.roi_size = 7

        # Two-layer MLP for dimensionality reduction
        # Input: 7*7*256 = 12544, Output: 512
        self.mlp = nn.Sequential(
            nn.Linear(7 * 7 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )

        # 2D Positional embeddings (learnable)
        # Max grid size assumption: 40x40 = 1600, but we use max_grid_cells=640
        self.pos_embed = nn.Parameter(torch.randn(max_grid_cells, 512))

        # Transformer encoder for grid cell relationships
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )

        # Classification head for OTSL tokens
        # 4 classes: C(0), L(1), U(2), X(3)
        self.classifier = nn.Linear(512, 4)

    def forward(self, x, grid_boxes):
        """
        Args:
            x: input images [B, 3, 960, 960]
            grid_boxes: list of B tensors, each of shape [R_i*C_i, 4]
                       boxes in (x1, y1, x2, y2) format in image coordinates

        Returns:
            predictions: list of B tensors, each of shape [R_i*C_i, 4] (logits)
        """
        # Extract features
        features = self.backbone(x)      # [B, 512, 30, 30]
        F_quarter = self.fpn(features)   # [B, 256, 240, 240]

        B = x.size(0)
        predictions = []

        for i in range(B):
            # Get grid boxes for this image
            boxes = grid_boxes[i]  # [R*C, 4]
            num_cells = boxes.size(0)

            if num_cells == 0:
                # No cells, return empty
                predictions.append(torch.zeros(0, 4, device=x.device))
                continue

            if num_cells > self.max_grid_cells:
                # Truncate if too many cells (shouldn't happen in practice)
                boxes = boxes[:self.max_grid_cells]
                num_cells = self.max_grid_cells

            # Prepare boxes for RoIAlign
            # RoIAlign expects boxes in [batch_idx, x1, y1, x2, y2] format
            # But since we process one image at a time, we use different approach
            batch_indices = torch.full((num_cells, 1), i, dtype=torch.float32, device=boxes.device)
            roi_boxes = torch.cat([batch_indices, boxes], dim=1)  # [R*C, 5]

            # Apply RoIAlign
            # Input feature map is F_quarter[i:i+1] which is [1, 256, 240, 240]
            # Boxes need to be scaled from image coords (960) to feature coords (240)
            scaled_boxes = boxes / 4.0  # Scale from 960 to 240
            roi_batch_indices = torch.zeros((num_cells, 1), dtype=torch.float32, device=boxes.device)
            roi_input = torch.cat([roi_batch_indices, scaled_boxes], dim=1)

            grid_features = roi_align(
                F_quarter[i:i+1],  # [1, 256, 240, 240]
                roi_input,         # [R*C, 5]
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=1.0,  # Already scaled boxes
                aligned=True
            )  # [R*C, 256, 7, 7]

            # Flatten and pass through MLP
            grid_features = grid_features.view(num_cells, -1)  # [R*C, 12544]
            grid_features = self.mlp(grid_features)  # [R*C, 512]

            # Add positional embeddings
            grid_features = grid_features + self.pos_embed[:num_cells]

            # Pass through Transformer
            grid_features = self.transformer(grid_features.unsqueeze(0)).squeeze(0)  # [R*C, 512]

            # Classify
            logits = self.classifier(grid_features)  # [R*C, 4]
            predictions.append(logits)

        return predictions


def focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    Focal loss for multi-class classification

    Args:
        predictions: [N, 4] logits
        targets: [N] class indices
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()


class MergeDataset(Dataset):
    def __init__(self, hf_dataset, split_model=None, device='cuda'):
        """
        Args:
            hf_dataset: HuggingFace dataset with OTSL annotations
            split_model: trained Split model to generate grids (if None, use ground truth)
            device: device for split model inference
        """
        self.hf_dataset = hf_dataset
        self.split_model = split_model
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((960, 960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if split_model is not None:
            self.split_model.eval()

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Load and transform image
        image = item['image'].convert('RGB')
        orig_width, orig_height = image.size
        image_transformed = self.transform(image)

        # Get OTSL for ground truth labels
        otsl = item['otsl']

        # Generate or use ground truth grid
        if self.split_model is not None:
            # Use Split model to generate grid
            with torch.no_grad():
                h_pred, v_pred = self.split_model(image_transformed.unsqueeze(0).to(self.device))
                h_pred = h_pred.squeeze(0).cpu()
                v_pred = v_pred.squeeze(0).cpu()

                # Convert to split positions
                h_splits = torch.where(h_pred > 0.5)[0].tolist()
                v_splits = torch.where(v_pred > 0.5)[0].tolist()
        else:
            # Use ground truth splits from actual cell bboxes
            cells_flat = item['cells'][0]
            
            # Parse OTSL to build 2D grid (same logic as split model)
            grid = []
            current_row = []
            cell_idx = 0
            
            for token in otsl:
                if token == 'nl':
                    if current_row:
                        grid.append(current_row)
                        current_row = []
                elif token == 'fcel' or token == 'ecel':
                    current_row.append({'type': token, 'cell_idx': cell_idx})
                    cell_idx += 1
                elif token in ['lcel', 'ucel', 'xcel']:
                    current_row.append({'type': token, 'cell_idx': None})
            
            if current_row:
                grid.append(current_row)
            
            # Derive row splits
            row_splits = []
            for row in grid:
                cell_indices = [item['cell_idx'] for item in row if item['cell_idx'] is not None]
                if cell_indices:
                    max_y = max(cells_flat[i]['bbox'][3] for i in cell_indices)
                    row_splits.append(max_y)
            
            # Derive column splits (skip horizontally merged cells)
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
            
            # Scale to 960x960
            h_splits = [int(y * 960 / orig_height) for y in row_splits]
            v_splits = [int(x * 960 / orig_width) for x in col_splits]


        # Generate grid boxes from splits
        h_splits_with_bounds = [0] + h_splits + [960]
        v_splits_with_bounds = [0] + v_splits + [960]

        R = len(h_splits_with_bounds) - 1
        C = len(v_splits_with_bounds) - 1

        # Create grid cell boxes
        grid_boxes = []
        for i in range(R):
            for j in range(C):
                y1 = h_splits_with_bounds[i]
                y2 = h_splits_with_bounds[i + 1]
                x1 = v_splits_with_bounds[j]
                x2 = v_splits_with_bounds[j + 1]
                grid_boxes.append([x1, y1, x2, y2])

        grid_boxes = torch.tensor(grid_boxes, dtype=torch.float32)

        # Get ground truth OTSL labels
        grid_labels = get_merge_ground_truth(otsl, R, C)
        grid_labels = grid_labels.view(-1)  # Flatten to [R*C]

        return image_transformed, grid_boxes, grid_labels


def test_single_image_gradient(model, dataset, device, idx=3):
    """
    Test on a single image to verify gradients are updating

    Args:
        model: MergeModel instance
        dataset: MergeDataset instance
        device: cuda/cpu
        idx: image index to test
    """
    print(f"\n{'='*60}")
    print(f"Testing gradient update on single image (index {idx})")
    print(f"{'='*60}\n")

    # Get single sample
    image, grid_boxes, grid_labels = dataset[idx]

    # Print info
    print(f"Image shape: {image.shape}")
    print(f"Grid boxes shape: {grid_boxes.shape}")
    print(f"Grid labels shape: {grid_labels.shape}")
    print(f"Number of grid cells: {grid_boxes.size(0)}")
    print(f"Label distribution: {torch.bincount(grid_labels)}")
    print(f"  Class 0 (C): {(grid_labels == 0).sum().item()}")
    print(f"  Class 1 (L): {(grid_labels == 1).sum().item()}")
    print(f"  Class 2 (U): {(grid_labels == 2).sum().item()}")
    print(f"  Class 3 (X): {(grid_labels == 3).sum().item()}")

    # Setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    # Move to device
    image = image.unsqueeze(0).to(device)
    grid_boxes = grid_boxes.to(device)
    grid_labels = grid_labels.to(device)

    # Save initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    print(f"\n--- Initial Forward Pass ---")
    # Forward pass
    predictions = model(image, [grid_boxes])
    pred = predictions[0]  # [R*C, 4]

    print(f"Predictions shape: {pred.shape}")
    print(f"Predictions range: [{pred.min():.3f}, {pred.max():.3f}]")

    # Calculate loss
    loss = focal_loss(pred, grid_labels, alpha=1.0, gamma=2.0)
    print(f"Initial loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    print(f"\n--- Gradient Check ---")
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            if grad_norm > 0.1:  # Only print significant gradients
                print(f"{name}: {grad_norm:.6f}")

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    # Update
    optimizer.step()

    print(f"\n--- After Optimizer Step ---")
    # Check parameter updates
    param_changes = {}
    for name, param in model.named_parameters():
        change = (param - initial_params[name]).norm().item()
        param_changes[name] = change
        if change > 1e-6:  # Only print changed parameters
            print(f"{name}: changed by {change:.6f}")

    # Forward pass again
    print(f"\n--- Second Forward Pass ---")
    predictions = model(image, [grid_boxes])
    pred = predictions[0]
    loss_after = focal_loss(pred, grid_labels, alpha=1.0, gamma=2.0)
    print(f"Loss after update: {loss_after.item():.6f}")
    print(f"Loss change: {loss_after.item() - loss.item():.6f}")

    # Predictions
    pred_classes = pred.argmax(dim=1)
    accuracy = (pred_classes == grid_labels).float().mean().item()
    print(f"Accuracy: {accuracy:.3f}")

    print(f"\n{'='*60}")
    print(f"Gradient test completed!")
    print(f"{'='*60}\n")

    # Verify gradients are flowing
    if sum(param_changes.values()) < 1e-5:
        print("WARNING: Parameters did not update! Check gradients.")
        return False
    else:
        print("SUCCESS: Gradients are flowing and parameters are updating!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TABLET Merge Model')
    parser.add_argument('--test-image-idx', type=int, default=3,
                        help='Image index to test gradient update')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    from datasets import load_dataset
    print("Loading FinTabNet dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Create dataset (using ground truth splits for now)
    dataset = MergeDataset(ds['train'])
    print(f"Dataset size: {len(dataset)}")

    # Create model
    model = MergeModel(max_grid_cells=640).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Test gradient update on single image
    success = test_single_image_gradient(model, dataset, device, idx=args.test_image_idx)

    if success:
        print("\n✓ Merge model is ready for training!")
    else:
        print("\n✗ Merge model has issues. Please debug.")

