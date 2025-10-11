"""
Evaluation script for TABLET split model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import logging

from split_model import SplitModel, TableDataset, focal_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
best_split_model_name = 'best_split_model.pth'

# Load dataset
dataset_name = "ds4sd/FinTabNet_OTSL"
dataset_name = "/home/ng6309/datascience/santhosh/datasets/PubTabNet_OTSL_full"
logger.info(f"Loading dataset {dataset_name}...")
ds = load_dataset(dataset_name)

# Use test split
#test_dataset = TableDataset(ds['test'])
test_dataset = TableDataset(ds['val'])

logger.info(f"📊 Test dataset size: {len(test_dataset):,}")

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Initialize model
model = SplitModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"📊 Model: {total_params:,} params")

# Load best model
logger.info(f"Loading model from {best_split_model_name}...")
checkpoint = torch.load(best_split_model_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
logger.info(f"Loaded model from epoch {checkpoint['epoch']}")


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


# Evaluation
logger.info("🚀 Starting evaluation...")
model.eval()
test_loss = 0
test_h_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
test_v_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count_error': 0}
test_batches = 0

with torch.no_grad():
    for images, h_gt_480, v_gt_480, h_gt_960, v_gt_960 in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        h_gt_480, v_gt_480 = h_gt_480.to(device), v_gt_480.to(device)
        h_gt_960, v_gt_960 = h_gt_960.to(device), v_gt_960.to(device)

        h_pred, v_pred = model(images)  # [B, 480]

        # Calculate losses at 480 resolution
        h_loss = focal_loss(h_pred, h_gt_480, alpha=1.0, gamma=2.0)
        v_loss = focal_loss(v_pred, v_gt_480, alpha=1.0, gamma=2.0)
        test_loss += (h_loss + v_loss).item()

        # Upsample predictions to 960 for metrics
        h_pred_960 = h_pred.repeat_interleave(2, dim=1)  # [B, 960]
        v_pred_960 = v_pred.repeat_interleave(2, dim=1)  # [B, 960]

        # Metrics at 960 resolution
        h_m = compute_metrics(h_pred_960, h_gt_960)
        v_m = compute_metrics(v_pred_960, v_gt_960)
        for k in ['precision', 'recall', 'f1', 'count_error']:
            test_h_metrics[k] += h_m[k]
            test_v_metrics[k] += v_m[k]
        test_batches += 1

avg_test_loss = test_loss / test_batches
for k in test_h_metrics:
    test_h_metrics[k] /= test_batches
    test_v_metrics[k] /= test_batches

logger.info(f"\n📊 Test Results:")
logger.info(f"  Test Loss: {avg_test_loss:.4f}")
logger.info(f"  H - F1: {test_h_metrics['f1']:.3f}, Precision: {test_h_metrics['precision']:.3f}, Recall: {test_h_metrics['recall']:.3f}, Count Error: {test_h_metrics['count_error']:.2f}")
logger.info(f"  V - F1: {test_v_metrics['f1']:.3f}, Precision: {test_v_metrics['precision']:.3f}, Recall: {test_v_metrics['recall']:.3f}, Count Error: {test_v_metrics['count_error']:.2f}")
avg_test_f1 = (test_h_metrics['f1'] + test_v_metrics['f1']) / 2
logger.info(f"  Avg F1: {avg_test_f1:.3f}")

logger.info("✅ Evaluation complete!")
