import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.ops import RoIAlign
from PIL import Image
import os
import numpy as np
from bs4 import BeautifulSoup
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

# Import split model to get grid structure
from split import SplitModel, get_ground_truth

def get_otsl_ground_truth(otsl_sequence):
    """
    Convert OTSL sequence to grid-based ground truth

    Args:
        otsl_sequence: list of OTSL tokens from dataset

    Returns:
        grid_labels: tensor of OTSL class indices for each grid cell (excluding 'nl')
    """
    # OTSL token to class mapping (excluding 'nl' as mentioned in paper)
    otsl_to_idx = {
        'fcel': 0,  # full cell (equivalent to 'C' in paper)
        'ecel': 1,  # empty cell (variant of 'C')
        'lcel': 2,  # left-looking cell (equivalent to 'L')
        'ucel': 3,  # up-looking cell (equivalent to 'U')
        # Note: 'nl' tokens are excluded as they're not used in split-merge grid-based approach
    }

    # Filter out 'nl' tokens and convert to indices
    grid_labels = []
    for token in otsl_sequence:
        if token != 'nl':
            if token in otsl_to_idx:
                grid_labels.append(otsl_to_idx[token])
            else:
                # Handle unknown tokens as empty cells
                grid_labels.append(1)  # ecel

    return torch.tensor(grid_labels, dtype=torch.long)

class BasicBlock(nn.Module):
    """Basic ResNet block"""
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

class StandardResNet18(nn.Module):
    """Standard ResNet-18 (not modified like in split model)"""
    def __init__(self):
        super().__init__()
        # Standard ResNet-18 architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (standard channels)
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

class FPNMerge(nn.Module):
    """Feature Pyramid Network for merge model outputting 256 channels at H/4×W/4"""
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
        """
        TABLET Merge Model for table structure recognition.

        Args:
            max_grid_cells: Maximum number of grid cells to handle (default: 640)
                           Increase for very large, dense tables if needed
        """
        super().__init__()
        self.backbone = StandardResNet18()
        self.fpn = FPNMerge()

        # RoIAlign for extracting grid cell features
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)

        # Two-layer MLP for dimensionality reduction
        self.mlp = nn.Sequential(
            nn.Linear(7 * 7 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # 2D positional embeddings for grid cells
        self.max_grid_cells = max_grid_cells
        self.pos_embed = nn.Parameter(torch.randn(max_grid_cells, 512))

        # Transformer encoder with correct dimensions
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )

        # Classification head for OTSL (4 classes: fcel, ecel, lcel, ucel)
        self.classifier = nn.Linear(512, 4)

    def forward(self, x, grid_boxes):
        """
        Args:
            x: Input images [B, 3, 960, 960]
            grid_boxes: List of tensors, each containing grid cell boxes for one image
                       Each tensor has shape [R*C, 4] where each row is [x1, y1, x2, y2]
        """
        batch_size = x.shape[0]

        # Extract features
        features = self.backbone(x)      # [B, 512, 30, 30]
        F_quarter = self.fpn(features)   # [B, 256, 240, 240] - This is F1/4

        all_grid_features = []
        all_sequence_lengths = []

        for b in range(batch_size):
            if len(grid_boxes) > b and grid_boxes[b] is not None:
                boxes = grid_boxes[b]  # [R*C, 4]
                num_cells = boxes.shape[0]

                if num_cells > 0:
                    # Prepare boxes for RoIAlign (add batch index)
                    batch_indices = torch.full((num_cells, 1), b, device=boxes.device, dtype=boxes.dtype)
                    roi_boxes = torch.cat([batch_indices, boxes], dim=1)  # [R*C, 5]

                    # Extract grid cell features using RoIAlign
                    grid_cell_features = self.roi_align(
                        F_quarter[b:b+1], roi_boxes
                    )  # [R*C, 256, 7, 7]

                    # Flatten and apply MLP
                    grid_cell_features = grid_cell_features.view(num_cells, -1)  # [R*C, 7*7*256]
                    grid_cell_features = self.mlp(grid_cell_features)  # [R*C, 512]

                    all_grid_features.append(grid_cell_features)
                    all_sequence_lengths.append(num_cells)
                else:
                    # Handle empty case
                    empty_features = torch.zeros((1, 512), device=x.device)
                    all_grid_features.append(empty_features)
                    all_sequence_lengths.append(1)
            else:
                # Handle missing grid case
                empty_features = torch.zeros((1, 512), device=x.device)
                all_grid_features.append(empty_features)
                all_sequence_lengths.append(1)

        # Pad sequences to same length (capped by max_grid_cells)
        max_length = min(max(all_sequence_lengths), self.max_grid_cells)
        if max(all_sequence_lengths) > self.max_grid_cells:
            print(f"Warning: Grid has {max(all_sequence_lengths)} cells, truncating to {self.max_grid_cells}")

        padded_features = torch.zeros((batch_size, max_length, 512), device=x.device)

        for b in range(batch_size):
            seq_len = min(all_sequence_lengths[b], max_length)
            if seq_len > 0:
                padded_features[b, :seq_len] = all_grid_features[b][:seq_len]

        # Add positional embeddings
        pos_embeddings = self.pos_embed[:max_length].unsqueeze(0).expand(batch_size, -1, -1)
        padded_features = padded_features + pos_embeddings

        # Create attention mask
        attention_mask = torch.zeros((batch_size, max_length), device=x.device, dtype=torch.bool)
        for b in range(batch_size):
            valid_len = min(all_sequence_lengths[b], max_length)
            attention_mask[b, valid_len:] = True

        # Apply transformer
        transformed_features = self.transformer(
            padded_features,
            src_key_padding_mask=attention_mask
        )  # [B, max_length, 512]

        # Classify each grid cell
        logits = self.classifier(transformed_features)  # [B, max_length, 4]

        return logits, all_sequence_lengths

def focal_loss(predictions, targets, alpha=1.0, gamma=2.0, ignore_index=-100):
    """Focal loss for OTSL classification"""
    ce_loss = F.cross_entropy(predictions, targets, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()

def create_grid_boxes(h_splits, v_splits, image_height=960, image_width=960, target_height=240, target_width=240):
    """
    Create grid cell bounding boxes from split predictions

    Args:
        h_splits: [H] binary array indicating horizontal split lines (from split model)
        v_splits: [W] binary array indicating vertical split lines (from split model)
        image_height, image_width: Original image dimensions (960x960)
        target_height, target_width: Target feature map dimensions for RoIAlign (H/4×W/4=240×240)

    Returns:
        boxes: [R*C, 4] tensor of grid cell boxes in target coordinates [x1, y1, x2, y2]
    """
    # Find row and column boundaries from split predictions
    # h_splits and v_splits should be [960] arrays from split model
    assert len(h_splits) == image_height, f"h_splits length {len(h_splits)} != image_height {image_height}"
    assert len(v_splits) == image_width, f"v_splits length {len(v_splits)} != image_width {image_width}"

    h_boundaries = [0]
    for i in range(len(h_splits)):
        if h_splits[i] > 0.5:  # Split line detected
            h_boundaries.append(i)
    h_boundaries.append(image_height)
    h_boundaries = sorted(list(set(h_boundaries)))

    v_boundaries = [0]
    for i in range(len(v_splits)):
        if v_splits[i] > 0.5:  # Split line detected
            v_boundaries.append(i)
    v_boundaries.append(image_width)
    v_boundaries = sorted(list(set(v_boundaries)))

    # Scale boundaries to target feature map dimensions (960×960 -> 240×240)
    h_scale = target_height / image_height  # 240/960 = 0.25
    v_scale = target_width / image_width    # 240/960 = 0.25

    boxes = []
    for i in range(len(h_boundaries) - 1):
        for j in range(len(v_boundaries) - 1):
            y1 = h_boundaries[i] * h_scale
            y2 = h_boundaries[i + 1] * h_scale
            x1 = v_boundaries[j] * v_scale
            x2 = v_boundaries[j + 1] * v_scale
            boxes.append([x1, y1, x2, y2])

    return torch.tensor(boxes, dtype=torch.float32)

class MergeDataset(Dataset):
    def __init__(self, hf_dataset, split_model=None, device='cuda'):
        self.hf_dataset = hf_dataset
        self.split_model = split_model
        self.device = device

        # Set split model to eval mode if provided
        if self.split_model is not None:
            self.split_model.eval()
            self.split_model.to(device)

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
        image_tensor = self.transform(image)

        # Get OTSL ground truth
        otsl_sequence = item['otsl']
        otsl_labels = get_otsl_ground_truth(otsl_sequence)

        # Get or generate grid structure
        if self.split_model is not None:
            # Use split model to generate grid
            with torch.no_grad():
                image_batch = image_tensor.unsqueeze(0).to(self.device)
                h_pred, v_pred = self.split_model(image_batch)
                h_splits = h_pred[0].cpu().numpy()  # [960] - horizontal split predictions
                v_splits = v_pred[0].cpu().numpy()  # [960] - vertical split predictions
        else:
            # Use ground truth splits (fallback)
            html_tags = item['html_restored']
            cells = item['cells']
            h_gt, v_gt = get_ground_truth(item['image'], html_tags, cells)
            h_splits = np.array(h_gt, dtype=np.float32)
            v_splits = np.array(v_gt, dtype=np.float32)

        # Create grid boxes
        grid_boxes = create_grid_boxes(h_splits, v_splits)

        return image_tensor, grid_boxes, otsl_labels

def collate_fn(batch):
    """Custom collate function to handle variable grid sizes"""
    images, grid_boxes_list, otsl_labels_list = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Keep grid_boxes as list since they have different sizes
    # Keep otsl_labels as list since they have different lengths

    return images, list(grid_boxes_list), list(otsl_labels_list)

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
    file_handler = logging.FileHandler(os.path.join(log_dir, 'merge_training.log'))
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
    writer = SummaryWriter(log_dir, comment='_TABLET_Merge_Model')

    # Log system information
    logger.info("=" * 60)
    logger.info("🚀 TABLET Merge Model Training Started")
    logger.info("=" * 60)
    logger.info(f"📁 Log directory: {log_dir}")
    logger.info(f"🔧 PyTorch version: {torch.__version__}")
    logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU device: {torch.cuda.get_device_name()}")
        logger.info(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return writer, logger

def evaluate_model(model, val_dataloader, device, logger=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    total_samples = 0

    if logger:
        logger.info("🔍 Starting validation evaluation...")

    with torch.no_grad():
        for batch_idx, (images, grid_boxes_list, otsl_labels_list) in enumerate(val_dataloader):
            images = images.to(device)

            # Move grid boxes to device
            grid_boxes_device = []
            for boxes in grid_boxes_list:
                if boxes is not None and len(boxes) > 0:
                    grid_boxes_device.append(boxes.to(device))
                else:
                    grid_boxes_device.append(None)

            logits, sequence_lengths = model(images, grid_boxes_device)

            # Calculate loss and accuracy
            batch_loss = 0
            batch_accuracy = 0
            batch_samples = 0

            for b in range(len(otsl_labels_list)):
                if len(otsl_labels_list[b]) > 0:
                    seq_len = min(len(otsl_labels_list[b]), logits.shape[1])
                    if seq_len > 0:
                        targets = otsl_labels_list[b][:seq_len].to(device)
                        preds = logits[b, :seq_len]

                        loss = focal_loss(preds, targets, alpha=1.0, gamma=2.0)
                        batch_loss += loss.item()

                        # Calculate accuracy
                        pred_classes = torch.argmax(preds, dim=-1)
                        accuracy = (pred_classes == targets).float().mean()
                        batch_accuracy += accuracy.item()
                        batch_samples += 1

            if batch_samples > 0:
                total_loss += batch_loss / batch_samples
                total_accuracy += batch_accuracy / batch_samples
                total_samples += batch_samples
                num_batches += 1

            # Log progress every 20 batches
            if logger and batch_idx % 20 == 0:
                logger.debug(f"Validation batch {batch_idx}/{len(val_dataloader)}")

    # Calculate averages
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
    else:
        avg_loss = float('inf')
        avg_accuracy = 0.0

    if logger:
        logger.info("✅ Validation completed")

    return {'loss': avg_loss, 'accuracy': avg_accuracy}

def train_merge_model(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging and tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/tablet_merge_{timestamp}"
    writer, logger = setup_logging(log_dir)

    logger.info(f"🎮 Using device: {device}")

    # Load split model if available
    split_model = None
    if args.split_model_path and os.path.exists(args.split_model_path):
        logger.info(f"📥 Loading split model from {args.split_model_path}")
        split_model = SplitModel().to(device)
        checkpoint = torch.load(args.split_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            split_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            split_model.load_state_dict(checkpoint)
        split_model.eval()
    else:
        logger.warning("⚠️  No split model provided, using ground truth splits")

    # Initialize merge model
    model = MergeModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Merge model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

    # Load dataset
    from datasets import load_dataset
    logger.info("📥 Loading FinTabNet dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Create train/val datasets
    train_dataset = MergeDataset(ds['train'], split_model, device)
    val_dataset = MergeDataset(ds['val'], split_model, device)

    # Subset for testing if specified
    if args.num_images > 0:
        logger.info(f"🔬 Using subset of {args.num_images} images for training")
        train_indices = list(range(min(args.num_images, len(train_dataset))))
        val_indices = list(range(min(args.num_images // 5, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    logger.info(f"📊 Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # Log dataset info to tensorboard
    writer.add_scalar('Dataset/Train_Size', len(train_dataset), 0)
    writer.add_scalar('Dataset/Val_Size', len(val_dataset), 0)
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)

    # Optimizer with polynomial decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-4
    )

    # Polynomial learning rate scheduler
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=args.epochs,
        power=0.9
    )

    # Log training configuration
    logger.info(f"📋 Training Configuration:")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ Batch Size: {args.batch_size}")
    logger.info(f"   ├─ Learning Rate: {args.lr}")
    logger.info(f"   ├─ Weight Decay: 5e-4")
    logger.info(f"   ├─ Gradient Clipping: 0.5")
    logger.info(f"   └─ LR Schedule: Polynomial (power=0.9)")

    # Training tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    global_step = 0
    start_time = datetime.now()

    logger.info("🚀 Starting training loop...")

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        logger.info(f"📖 Epoch {epoch+1}/{args.epochs} started")

        for batch_idx, (images, grid_boxes_list, otsl_labels_list) in enumerate(train_dataloader):
            images = images.to(device)

            # Move grid boxes to device
            grid_boxes_device = []
            for boxes in grid_boxes_list:
                if boxes is not None and len(boxes) > 0:
                    grid_boxes_device.append(boxes.to(device))
                else:
                    grid_boxes_device.append(None)

            optimizer.zero_grad()

            logits, sequence_lengths = model(images, grid_boxes_device)

            # Calculate loss for each sample in batch
            total_loss = 0
            total_accuracy = 0
            valid_samples = 0

            for b in range(len(otsl_labels_list)):
                if len(otsl_labels_list[b]) > 0:
                    seq_len = min(len(otsl_labels_list[b]), logits.shape[1])
                    if seq_len > 0:
                        targets = otsl_labels_list[b][:seq_len].to(device)
                        preds = logits[b, :seq_len]

                        loss = focal_loss(preds, targets, alpha=1.0, gamma=2.0)
                        total_loss += loss

                        # Calculate accuracy
                        pred_classes = torch.argmax(preds, dim=-1)
                        accuracy = (pred_classes == targets).float().mean()
                        total_accuracy += accuracy
                        valid_samples += 1

            if valid_samples > 0:
                batch_loss = total_loss / valid_samples
                batch_accuracy = total_accuracy / valid_samples

                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()

                # Track losses
                epoch_loss += batch_loss.item()
                epoch_accuracy += batch_accuracy.item()
                num_batches += 1

                # Enhanced tensorboard logging
                writer.add_scalar('Train/Loss_Step', batch_loss.item(), global_step)
                writer.add_scalar('Train/Accuracy_Step', batch_accuracy.item(), global_step)
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

            # Log every few batches
            if batch_idx % 10 == 0:
                elapsed = datetime.now() - epoch_start_time
                progress = (batch_idx + 1) / len(train_dataloader) * 100
                if valid_samples > 0:
                    log_msg = f'Epoch {epoch+1}/{args.epochs} [{progress:5.1f}%] Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {batch_loss:.4f}, Acc: {batch_accuracy:.3f} | Elapsed: {elapsed}'
                else:
                    log_msg = f'Epoch {epoch+1}/{args.epochs} [{progress:5.1f}%] Batch {batch_idx+1}/{len(train_dataloader)} | No valid samples | Elapsed: {elapsed}'
                logger.info(log_msg)

        # Update learning rate
        scheduler.step()

        # Epoch averages
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
        else:
            avg_loss = float('inf')
            avg_accuracy = 0.0

        # Validation
        epoch_duration = datetime.now() - epoch_start_time
        logger.info(f"🗓️ Epoch {epoch+1} training completed in {epoch_duration}")
        val_metrics = evaluate_model(model, val_dataloader, device, logger)

        # Comprehensive epoch logging
        logger.info(f"📊 Epoch {epoch+1}/{args.epochs} Results:")
        logger.info(f"   Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.3f}")
        logger.info(f"   Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.3f}")

        # Enhanced tensorboard logging
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy_Epoch', avg_accuracy, epoch)
        writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Accuracy_Epoch', val_metrics['accuracy'], epoch)

        # Save best model (based on accuracy)
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(log_dir, 'best_merge_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy,
            }, best_model_path)
            logger.info(f'✨ New best model saved! Accuracy: {best_val_accuracy:.4f}, Loss: {best_val_loss:.4f}')

        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f'merge_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)
            logger.info(f'💾 Checkpoint saved: {checkpoint_path}')

    writer.close()

    # Training completion
    total_training_time = datetime.now() - start_time
    logger.info("="*60)
    logger.info("🏆 Merge Model Training Completed Successfully!")
    logger.info(f"⏱️ Total training time: {total_training_time}")
    logger.info(f"🎆 Best validation accuracy: {best_val_accuracy:.4f}")
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train TABLET Merge Model')

    # Dataset arguments
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Number of images to train on (-1 for full dataset)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=24,
                        help='Number of epochs to train (default: 24)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')

    # Model arguments
    parser.add_argument('--split-model-path', type=str, default=None,
                        help='Path to trained split model (optional)')

    # Checkpointing arguments
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("🚀 Starting TABLET Merge Model Training")
    print(f"📊 Config: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    if args.split_model_path:
        print(f"🔗 Using split model: {args.split_model_path}")
    else:
        print("⚠️  No split model provided, using ground truth splits")
    if args.num_images > 0:
        print(f"🔬 Testing mode: Training on {args.num_images} images only")
    else:
        print("🎯 Full training mode: Using complete dataset")

    print("📈 Tensorboard logging enabled")

    # Train the model
    model, log_dir = train_merge_model(args)

    print(f"✅ Training completed! Logs and models saved in: {log_dir}")
    print(f"📂 Best model: {log_dir}/best_merge_model.pth")
    print(f"📂 Final model: {log_dir}/final_merge_model.pth")

    print(f"📈 View tensorboard: tensorboard --logdir {log_dir}")