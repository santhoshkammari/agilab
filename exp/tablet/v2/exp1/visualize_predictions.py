"""
Visualize predictions from trained model
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from split_model import SplitModel, TableDataset
from PIL import Image, ImageDraw
import numpy as np

def visualize_prediction(model, dataset, idx, device, threshold=0.5):
    """Visualize prediction for a single image"""
    model.eval()

    # Get sample
    image, h_target, v_target = dataset[idx]

    # Get original PIL image
    original_image = dataset.hf_dataset[idx]['image']
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # Predict
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        h_pred, v_pred = model(image_batch)
        h_pred = h_pred.squeeze(0).cpu()
        v_pred = v_pred.squeeze(0).cpu()

    # Apply threshold
    h_binary = (h_pred > threshold).float()
    v_binary = (v_pred > threshold).float()

    # Count predictions
    h_splits = h_binary.sum().item()
    v_splits = v_binary.sum().item()
    h_gt_splits = h_target.sum().item()
    v_gt_splits = v_target.sum().item()

    # Calculate meaningful metrics for split detection
    # Precision: Of all predicted splits, how many are correct?
    # Recall: Of all ground truth splits, how many did we find?
    # F1: Harmonic mean of precision and recall

    def compute_metrics(pred, target):
        """Compute Precision, Recall, F1 for binary vectors"""
        tp = (pred * target).sum().item()  # True positives
        fp = (pred * (1 - target)).sum().item()  # False positives
        fn = ((1 - pred) * target).sum().item()  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Count error: absolute difference in number of splits
        count_error = abs(pred.sum().item() - target.sum().item())

        return precision, recall, f1, count_error

    h_prec, h_rec, h_f1, h_count_err = compute_metrics(h_binary, h_target)
    v_prec, v_rec, v_f1, v_count_err = compute_metrics(v_binary, v_target)

    print(f"\nImage {idx}:")
    print(f"  H Splits: {h_splits:.0f} pred / {h_gt_splits:.0f} GT | Error: {h_count_err:.0f}")
    print(f"    Precision: {h_prec:.3f} | Recall: {h_rec:.3f} | F1: {h_f1:.3f}")
    print(f"  V Splits: {v_splits:.0f} pred / {v_gt_splits:.0f} GT | Error: {v_count_err:.0f}")
    print(f"    Precision: {v_prec:.3f} | Recall: {v_rec:.3f} | F1: {v_f1:.3f}")
    print(f"  H pred range: [{h_pred.min():.3f}, {h_pred.max():.3f}]")
    print(f"  V pred range: [{v_pred.min():.3f}, {v_pred.max():.3f}]")

    # Visualize
    W, H = original_image.size

    # Calculate grid stats
    num_rows = int(h_target.sum().item()) + 1  # splits + 1 = cells
    num_cols = int(v_target.sum().item()) + 1
    pred_rows = int(h_binary.sum().item()) + 1
    pred_cols = int(v_binary.sum().item()) + 1

    # Info panel dimensions
    info_height = 150
    info_width = 450
    padding = 30

    try:
        from PIL import ImageFont
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font_title = None
        font_text = None

    # Create top-bottom visualization with info panels on the left
    total_width = info_width + W + padding * 3
    total_height = H * 2 + info_height * 2 + padding * 5
    vis_image = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(vis_image)

    # Top section: Ground Truth
    y_offset = padding

    # GT Info Panel
    info_x = padding
    info_y = y_offset

    draw.rectangle([info_x, info_y, info_x + info_width, info_y + info_height],
                   fill='#e8f5e9', outline='#2e7d32', width=3)

    draw.text((info_x + 15, info_y + 15), "GROUND TRUTH", fill='#1b5e20', font=font_title)
    draw.text((info_x + 15, info_y + 55), f"Rows: {num_rows}", fill='black', font=font_text)
    draw.text((info_x + 15, info_y + 80), f"Columns: {num_cols}", fill='black', font=font_text)
    draw.text((info_x + 15, info_y + 105), f"Total Cells: {num_rows * num_cols}", fill='black', font=font_text)
    draw.text((info_x + 220, info_y + 55), f"H Splits: {int(h_gt_splits)}", fill='#c62828', font=font_text)
    draw.text((info_x + 220, info_y + 80), f"V Splits: {int(v_gt_splits)}", fill='#1565c0', font=font_text)

    # GT Image
    gt_image = original_image.copy()
    draw_gt = ImageDraw.Draw(gt_image)

    # Draw GT horizontal lines (red, thicker)
    for y in range(960):
        if h_target[y] == 1:
            y_scaled = int(y * H / 960)
            draw_gt.line([(0, y_scaled), (W, y_scaled)], fill='#ff0000', width=4)

    # Draw GT vertical lines (blue, thicker)
    for x in range(960):
        if v_target[x] == 1:
            x_scaled = int(x * W / 960)
            draw_gt.line([(x_scaled, 0), (x_scaled, H)], fill='#0000ff', width=4)

    vis_image.paste(gt_image, (info_width + padding * 2, y_offset))

    # Bottom section: Predictions
    y_offset = H + info_height + padding * 3

    # Pred Info Panel
    info_y = y_offset

    draw.rectangle([info_x, info_y, info_x + info_width, info_y + info_height],
                   fill='#e3f2fd', outline='#1565c0', width=3)

    draw.text((info_x + 15, info_y + 15), "PREDICTED", fill='#0d47a1', font=font_title)
    draw.text((info_x + 15, info_y + 55), f"Rows: {pred_rows}", fill='black', font=font_text)
    draw.text((info_x + 15, info_y + 80), f"Columns: {pred_cols}", fill='black', font=font_text)
    draw.text((info_x + 15, info_y + 105), f"Total Cells: {pred_rows * pred_cols}", fill='black', font=font_text)
    draw.text((info_x + 220, info_y + 55), f"H Splits: {int(h_splits)} (Err: {int(h_count_err)})", fill='#c62828', font=font_text)
    draw.text((info_x + 220, info_y + 80), f"V Splits: {int(v_splits)} (Err: {int(v_count_err)})", fill='#1565c0', font=font_text)
    draw.text((info_x + 220, info_y + 105), f"H F1: {h_f1:.2%} | V F1: {v_f1:.2%}", fill='black', font=font_text)

    # Pred Image
    pred_image = original_image.copy()
    draw_pred = ImageDraw.Draw(pred_image)

    # Draw predicted horizontal lines (red, thicker)
    for y in range(960):
        if h_binary[y] == 1:
            y_scaled = int(y * H / 960)
            draw_pred.line([(0, y_scaled), (W, y_scaled)], fill='#ff0000', width=4)

    # Draw predicted vertical lines (blue, thicker)
    for x in range(960):
        if v_binary[x] == 1:
            x_scaled = int(x * W / 960)
            draw_pred.line([(x_scaled, 0), (x_scaled, H)], fill='#0000ff', width=4)

    vis_image.paste(pred_image, (info_width + padding * 2, y_offset))

    # Save
    output_path = f'prediction_vis_{idx}.png'
    vis_image.save(output_path)
    print(f"  Saved: {output_path}")

    return h_f1, v_f1, h_count_err, v_count_err

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")
    val_dataset = TableDataset(ds['val'])

    # Load model
    print("Loading model...")
    model = SplitModel().to(device)

    # Try to find the best model - check multiple possible locations
    import os
    possible_paths = [
        'best_split_model.pth',  # Current directory (from train_split_fixed.py)
        '/home/ng6309/datascience/santhosh/experiments/tablet/best_split_model.pth',
        '/home/ng6309/datascience/santhosh/experiments/tablet/runs/tablet_split_20251006_214335/best_split_model.pth',
    ]

    checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("ERROR: No trained model found! Please train the model first using train_split_fixed.py")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nModel trained for {checkpoint['epoch']} epochs")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    print(f"Val H F1: {checkpoint['val_h_f1']:.3f}")
    print(f"Val V F1: {checkpoint['val_v_f1']:.3f}")

    # Visualize first 5 validation images
    print("\n" + "="*60)
    print("Visualizing predictions (Top: Ground Truth, Bottom: Predicted)")
    print("="*60)

    total_h_f1 = 0
    total_v_f1 = 0
    total_h_err = 0
    total_v_err = 0
    num_samples = 5

    for idx in range(num_samples):
        h_f1, v_f1, h_err, v_err = visualize_prediction(model, val_dataset, idx, device)
        total_h_f1 += h_f1
        total_v_f1 += v_f1
        total_h_err += h_err
        total_v_err += v_err

    print("\n" + "="*60)
    print(f"Average over {num_samples} samples:")
    print(f"  H F1: {total_h_f1/num_samples:.3f} | Avg Count Error: {total_h_err/num_samples:.1f}")
    print(f"  V F1: {total_v_f1/num_samples:.3f} | Avg Count Error: {total_v_err/num_samples:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()
