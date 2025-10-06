"""
Visualize predictions from trained model
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from final_model import SplitModel, TableDataset
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

    # Calculate metrics
    h_acc = (h_binary == h_target).float().mean().item()
    v_acc = (v_binary == v_target).float().mean().item()

    # Count predictions
    h_splits = h_binary.sum().item()
    v_splits = v_binary.sum().item()
    h_gt_splits = h_target.sum().item()
    v_gt_splits = v_target.sum().item()

    print(f"\nImage {idx}:")
    print(f"  H: {h_splits:.0f} predicted / {h_gt_splits:.0f} GT | Acc: {h_acc:.3f}")
    print(f"  V: {v_splits:.0f} predicted / {v_gt_splits:.0f} GT | Acc: {v_acc:.3f}")
    print(f"  H pred range: [{h_pred.min():.3f}, {h_pred.max():.3f}]")
    print(f"  V pred range: [{v_pred.min():.3f}, {v_pred.max():.3f}]")

    # Visualize
    W, H = original_image.size

    # Create side-by-side visualization
    vis_width = W * 2
    vis_image = Image.new('RGB', (vis_width, H))

    # Left: Ground truth
    gt_image = original_image.copy()
    draw_gt = ImageDraw.Draw(gt_image)

    # Draw GT horizontal lines (red)
    for y in range(960):
        if h_target[y] == 1:
            y_scaled = int(y * H / 960)
            draw_gt.line([(0, y_scaled), (W, y_scaled)], fill='red', width=2)

    # Draw GT vertical lines (blue)
    for x in range(960):
        if v_target[x] == 1:
            x_scaled = int(x * W / 960)
            draw_gt.line([(x_scaled, 0), (x_scaled, H)], fill='blue', width=2)

    # Right: Predictions
    pred_image = original_image.copy()
    draw_pred = ImageDraw.Draw(pred_image)

    # Draw predicted horizontal lines (red)
    for y in range(960):
        if h_binary[y] == 1:
            y_scaled = int(y * H / 960)
            draw_pred.line([(0, y_scaled), (W, y_scaled)], fill='red', width=2)

    # Draw predicted vertical lines (blue)
    for x in range(960):
        if v_binary[x] == 1:
            x_scaled = int(x * W / 960)
            draw_pred.line([(x_scaled, 0), (x_scaled, H)], fill='blue', width=2)

    # Combine
    vis_image.paste(gt_image, (0, 0))
    vis_image.paste(pred_image, (W, 0))

    # Save
    output_path = f'prediction_vis_{idx}.png'
    vis_image.save(output_path)
    print(f"  Saved: {output_path}")

    return h_acc, v_acc

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
    checkpoint = torch.load('best_model_full.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nModel trained for {checkpoint['epoch']} epochs")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    print(f"Val H_acc: {checkpoint['val_h_acc']:.3f}")
    print(f"Val V_acc: {checkpoint['val_v_acc']:.3f}")

    # Visualize first 5 validation images
    print("\n" + "="*60)
    print("Visualizing predictions (Left: Ground Truth, Right: Predicted)")
    print("="*60)

    total_h_acc = 0
    total_v_acc = 0
    num_samples = 5

    for idx in range(num_samples):
        h_acc, v_acc = visualize_prediction(model, val_dataset, idx, device)
        total_h_acc += h_acc
        total_v_acc += v_acc

    print("\n" + "="*60)
    print(f"Average over {num_samples} samples:")
    print(f"  H_acc: {total_h_acc/num_samples:.3f}")
    print(f"  V_acc: {total_v_acc/num_samples:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()

