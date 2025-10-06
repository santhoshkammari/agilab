"""
Visualize predictions from trained merge model
Similar to visualize_predictions.py for split model
"""
import torch
from datasets import load_dataset
from merge_model import MergeModel, MergeDataset, get_merge_ground_truth
from final_model import SplitModel
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def visualize_merge_prediction(merge_model, split_model, dataset, idx, device, threshold=0.5):
    """
    Visualize merge model prediction for a single image

    Args:
        merge_model: trained MergeModel
        split_model: trained SplitModel (for generating grid)
        dataset: MergeDataset
        idx: image index
        device: cuda/cpu
        threshold: threshold for split model (not used for merge predictions)
    """
    merge_model.eval()
    if split_model is not None:
        split_model.eval()

    # Get sample
    image, grid_boxes, grid_labels_gt = dataset[idx]

    # Get original PIL image
    original_image = dataset.hf_dataset[idx]['image'].convert('RGB')
    original_image = original_image.resize((960, 960))

    # Predict
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        grid_boxes_batch = grid_boxes.to(device)

        # Merge model prediction
        predictions = merge_model(image_batch, [grid_boxes_batch])
        pred_logits = predictions[0].cpu()  # [R*C, 4]

        # Get predicted classes
        pred_classes = pred_logits.argmax(dim=1)

    # Calculate accuracy
    accuracy = (pred_classes == grid_labels_gt).float().mean().item()

    # Count per class
    token_names = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}
    token_colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple'}

    print(f"\n{'='*60}")
    print(f"Image {idx}:")
    print(f"  Grid cells: {grid_boxes.shape[0]}")
    print(f"  Overall accuracy: {accuracy:.3f}")
    print(grid_labels_gt)

    print(f"\n  Ground Truth Distribution:")
    for i in range(4):
        count_gt = (grid_labels_gt == i).sum().item()
        count_pred = (pred_classes == i).sum().item()
        print(f"    Class {i} ({token_names[i]}): GT={count_gt:3d}, Pred={count_pred:3d}")

    # Per-class accuracy
    print(f"\n  Per-class Accuracy:")
    for i in range(4):
        mask = grid_labels_gt == i
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == grid_labels_gt[mask]).float().mean().item()
            print(f"    Class {i} ({token_names[i]}): {class_acc:.3f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Ground Truth
    img_gt = original_image.copy()
    draw_gt = ImageDraw.Draw(img_gt)

    for i, box in enumerate(grid_boxes):
        x1, y1, x2, y2 = box.tolist()
        label = grid_labels_gt[i].item()
        color = token_colors[label]

        draw_gt.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label text
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        text = token_names[label]
        draw_gt.text((cx, cy), text, fill=color)

    axes[0].imshow(img_gt)
    axes[0].set_title('Ground Truth OTSL\nGreen=C, Blue=L, Red=U, Purple=X', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Right: Predictions
    img_pred = original_image.copy()
    draw_pred = ImageDraw.Draw(img_pred)

    correct = 0
    incorrect = 0

    for i, box in enumerate(grid_boxes):
        x1, y1, x2, y2 = box.tolist()
        label_pred = pred_classes[i].item()
        label_gt = grid_labels_gt[i].item()

        is_correct = (label_pred == label_gt)

        if is_correct:
            color = token_colors[label_pred]
            width = 3
            correct += 1
        else:
            color = 'orange'  # Wrong predictions in orange
            width = 2
            incorrect += 1

        draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Draw label text
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        text = token_names[label_pred]
        if not is_correct:
            text += f'!{token_names[label_gt]}'  # Show GT for wrong predictions
        draw_pred.text((cx, cy), text, fill=color if is_correct else 'red')

    axes[1].imshow(img_pred)
    axes[1].set_title(f'Predictions\nAcc: {accuracy:.1%} ({correct}/{len(grid_boxes)})\nOrange = Wrong',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    output_path = f'merge_prediction_vis_{idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")

    return accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Merge Model Predictions')
    parser.add_argument('--model-path', type=str, default='best_merge_model.pth',
                        help='Path to trained merge model')
    parser.add_argument('--split-model-path', type=str, default='best_model_full.pth',
                        help='Path to trained split model (optional)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--use-split-model', action='store_true',
                        help='Use split model to generate grids (instead of GT)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    # Load split model if requested
    split_model = None
    if args.use_split_model:
        try:
            print(f"Loading split model from {args.split_model_path}...")
            split_model = SplitModel().to(device)
            checkpoint = torch.load(args.split_model_path, map_location=device)
            split_model.load_state_dict(checkpoint['model_state_dict'])
            split_model.eval()
            print(f"  Split model loaded (epoch {checkpoint['epoch']})")
        except FileNotFoundError:
            print(f"  Warning: Split model not found, using ground truth grids")
            split_model = None

    # Create dataset
    val_dataset = MergeDataset(ds['val'], split_model=split_model, device=device)

    # Load merge model
    print(f"\nLoading merge model from {args.model_path}...")
    merge_model = MergeModel().to(device)

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        merge_model.load_state_dict(checkpoint['model_state_dict'])

        print(f"  Model trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"  Val loss: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"  Val accuracy: {checkpoint['val_accuracy']:.3f}")
    except FileNotFoundError:
        print(f"  Warning: Model file not found, using random initialization")

    # Visualize predictions
    print(f"\n{'='*60}")
    print("Visualizing Merge Model Predictions")
    print("(Left: Ground Truth, Right: Predicted)")
    print(f"{'='*60}")

    total_acc = 0
    num_samples = min(args.num_samples, len(val_dataset))

    for idx in range(num_samples):
        acc = visualize_merge_prediction(merge_model, split_model, val_dataset, idx, device)
        total_acc += acc

    print(f"\n{'='*60}")
    print(f"Average over {num_samples} samples:")
    print(f"  Accuracy: {total_acc/num_samples:.3f}")
    print(f"{'='*60}")

    print(f"\n✓ Visualizations saved as merge_prediction_vis_*.png")


if __name__ == "__main__":
    main()

