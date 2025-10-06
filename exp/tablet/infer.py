"""
TABLET Inference Script

Takes a table image and produces:
1. Grid split predictions (using split model)
2. Merge predictions (using merge model)
3. Outputs visualization image and JSON results

Usage:
    python infer.py --image path/to/table.png
    python infer.py --image path/to/table.png --split-model path/to/split.pth --merge-model path/to/merge.pth
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
from pathlib import Path
import torchvision.transforms as transforms

# Import model architectures
from final_model import SplitModel
from merge_model import MergeModel


def load_and_preprocess_image(image_path):
    """Load and preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = image.size

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((960, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    return image, image_tensor, orig_width, orig_height


def predict_splits(split_model, image_tensor, device, threshold=0.5):
    """
    Run split model to get horizontal and vertical split positions

    Returns:
        h_splits: list of y-coordinates for horizontal splits
        v_splits: list of x-coordinates for vertical splits
    """
    split_model.eval()

    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        h_pred, v_pred = split_model(image_batch)
        h_pred = h_pred.squeeze(0).cpu().numpy()  # [960]
        v_pred = v_pred.squeeze(0).cpu().numpy()  # [960]

    # Convert to binary and extract split positions
    h_binary = (h_pred > threshold).astype(int)
    v_binary = (v_pred > threshold).astype(int)

    # Find split positions (where binary == 1)
    h_splits = np.where(h_binary == 1)[0].tolist()
    v_splits = np.where(v_binary == 1)[0].tolist()

    # Group consecutive pixels into single splits (take middle)
    def group_consecutive(splits):
        if not splits:
            return []
        groups = []
        current_group = [splits[0]]

        for i in range(1, len(splits)):
            if splits[i] - splits[i-1] <= 1:
                current_group.append(splits[i])
            else:
                groups.append(int(np.mean(current_group)))
                current_group = [splits[i]]

        groups.append(int(np.mean(current_group)))
        return groups

    h_splits = group_consecutive(h_splits)
    v_splits = group_consecutive(v_splits)

    return h_splits, v_splits, h_pred, v_pred


def create_grid_boxes(h_splits, v_splits):
    """
    Create grid cell bounding boxes from split positions

    Returns:
        grid_boxes: tensor [R*C, 4] in (x1, y1, x2, y2) format
        R: number of rows
        C: number of columns
    """
    h_splits_with_bounds = [0] + h_splits + [960]
    v_splits_with_bounds = [0] + v_splits + [960]

    R = len(h_splits_with_bounds) - 1
    C = len(v_splits_with_bounds) - 1

    grid_boxes = []
    for i in range(R):
        for j in range(C):
            y1 = h_splits_with_bounds[i]
            y2 = h_splits_with_bounds[i + 1]
            x1 = v_splits_with_bounds[j]
            x2 = v_splits_with_bounds[j + 1]
            grid_boxes.append([x1, y1, x2, y2])

    grid_boxes = torch.tensor(grid_boxes, dtype=torch.float32)
    return grid_boxes, R, C


def predict_merges(merge_model, image_tensor, grid_boxes, device):
    """
    Run merge model to classify each grid cell

    Returns:
        predictions: tensor [R*C, 4] class probabilities
        pred_classes: tensor [R*C] predicted class indices (0=C, 1=L, 2=U, 3=X)
    """
    merge_model.eval()

    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        grid_boxes_batch = [grid_boxes.to(device)]
        predictions = merge_model(image_batch, grid_boxes_batch)

        # Get predictions for first (and only) image
        pred_logits = predictions[0]  # [R*C, 4]
        pred_probs = F.softmax(pred_logits, dim=1)  # Convert to probabilities
        pred_classes = pred_logits.argmax(dim=1)  # [R*C]

    return pred_probs.cpu(), pred_classes.cpu()


def visualize_results(image, h_splits, v_splits, grid_boxes, pred_classes, R, C):
    """
    Create visualization with grid lines and merge labels

    Returns:
        PIL Image with visualization
    """
    # Resize image to 960x960 for visualization
    img_vis = image.resize((960, 960))
    draw = ImageDraw.Draw(img_vis)

    # Color map for classes
    class_colors = {
        0: 'lightgreen',  # C (new cell)
        1: 'lightblue',   # L (merge left)
        2: 'lightyellow', # U (merge up)
        3: 'lightcoral'   # X (merge both)
    }

    class_names = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}

    # Draw grid cells with class colors
    for idx, (box, cls) in enumerate(zip(grid_boxes, pred_classes)):
        x1, y1, x2, y2 = box.tolist()
        color = class_colors[cls.item()]

        # Fill cell with semi-transparent color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw class label in center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        label = class_names[cls.item()]
        draw.text((cx, cy), label, fill='black')

    # Draw split lines
    for y in h_splits:
        draw.line([(0, y), (960, y)], fill='red', width=2)

    for x in v_splits:
        draw.line([(x, 0), (x, 960)], fill='red', width=2)

    return img_vis


def create_json_output(h_splits, v_splits, grid_boxes, pred_classes, pred_probs, R, C, orig_width, orig_height):
    """
    Create JSON output with all results

    Returns:
        dict with structure and cell information
    """
    class_names = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}

    # Scale splits back to original image coordinates
    scale_x = orig_width / 960
    scale_y = orig_height / 960

    h_splits_orig = [int(y * scale_y) for y in h_splits]
    v_splits_orig = [int(x * scale_x) for x in v_splits]

    # Build cell data
    cells = []
    for idx, (box, cls, probs) in enumerate(zip(grid_boxes, pred_classes, pred_probs)):
        x1, y1, x2, y2 = box.tolist()

        # Scale box to original coordinates
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        # Compute row and column indices
        row_idx = idx // C
        col_idx = idx % C

        cell_data = {
            'cell_id': idx,
            'row': row_idx,
            'col': col_idx,
            'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
            'bbox_960': [int(x1), int(y1), int(x2), int(y2)],
            'class': class_names[cls.item()],
            'class_id': cls.item(),
            'probabilities': {
                'C': float(probs[0]),
                'L': float(probs[1]),
                'U': float(probs[2]),
                'X': float(probs[3])
            }
        }
        cells.append(cell_data)

    output = {
        'image_size': {
            'original': {'width': orig_width, 'height': orig_height},
            'processed': {'width': 960, 'height': 960}
        },
        'grid_structure': {
            'rows': R,
            'cols': C,
            'total_cells': R * C
        },
        'splits': {
            'horizontal': h_splits_orig,
            'vertical': v_splits_orig,
            'horizontal_960': h_splits,
            'vertical_960': v_splits
        },
        'cells': cells
    }

    return output


def main():
    parser = argparse.ArgumentParser(description='TABLET Merge Inference')

    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input table image')

    # Model paths
    parser.add_argument('--split-model', type=str,
                        default='best_model_full.pth',
                        help='Path to trained split model')
    parser.add_argument('--merge-model', type=str,
                        default='runs/tablet_merge_20251006_205547/best_merge_model.pth',
                        help='Path to trained merge model')

    # Output paths
    parser.add_argument('--output-image', type=str, default='output.png',
                        help='Path to save visualization image')
    parser.add_argument('--output-json', type=str, default='res.json',
                        help='Path to save JSON results')

    # Inference parameters
    parser.add_argument('--split-threshold', type=float, default=0.5,
                        help='Threshold for split detection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return

    print(f"\n{'='*60}")
    print(f"TABLET Merge Inference")
    print(f"{'='*60}")
    print(f"Input image: {args.image}")
    print(f"Split model: {args.split_model}")
    print(f"Merge model: {args.merge_model}")
    print(f"Output image: {args.output_image}")
    print(f"Output JSON: {args.output_json}")
    print(f"{'='*60}\n")

    # Load image
    print("Loading and preprocessing image...")
    image, image_tensor, orig_width, orig_height = load_and_preprocess_image(args.image)
    print(f"  Original size: {orig_width}x{orig_height}")
    print(f"  Processed size: 960x960")

    # Load split model
    print(f"\nLoading split model from {args.split_model}...")
    split_model = SplitModel().to(device)
    try:
        checkpoint = torch.load(args.split_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            split_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        else:
            split_model.load_state_dict(checkpoint)
            print(f"  Loaded model weights")
    except FileNotFoundError:
        print(f"  Warning: Split model not found at {args.split_model}")
        return

    # Load merge model
    print(f"\nLoading merge model from {args.merge_model}...")
    merge_model = MergeModel(max_grid_cells=640).to(device)
    try:
        checkpoint = torch.load(args.merge_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            merge_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?') + 1}")
            if 'val_accuracy' in checkpoint:
                print(f"  Validation accuracy: {checkpoint['val_accuracy']:.3f}")
        else:
            merge_model.load_state_dict(checkpoint)
            print(f"  Loaded model weights")
    except FileNotFoundError:
        print(f"  Warning: Merge model not found at {args.merge_model}")
        return

    # Step 1: Predict splits
    print("\n[1/3] Predicting grid splits...")
    h_splits, v_splits, h_pred, v_pred = predict_splits(
        split_model, image_tensor, device, threshold=args.split_threshold
    )
    print(f"  Found {len(h_splits)} horizontal splits")
    print(f"  Found {len(v_splits)} vertical splits")

    # Step 2: Create grid boxes
    print("\n[2/3] Creating grid structure...")
    grid_boxes, R, C = create_grid_boxes(h_splits, v_splits)
    print(f"  Grid size: {R} rows × {C} cols = {R*C} cells")

    # Step 3: Predict merges
    print("\n[3/3] Predicting cell merges...")
    pred_probs, pred_classes = predict_merges(merge_model, image_tensor, grid_boxes, device)

    # Count classes
    class_counts = {
        'C': (pred_classes == 0).sum().item(),
        'L': (pred_classes == 1).sum().item(),
        'U': (pred_classes == 2).sum().item(),
        'X': (pred_classes == 3).sum().item()
    }
    print(f"  Class distribution:")
    print(f"    C (new cell): {class_counts['C']}")
    print(f"    L (merge left): {class_counts['L']}")
    print(f"    U (merge up): {class_counts['U']}")
    print(f"    X (merge both): {class_counts['X']}")

    # Create visualization
    print(f"\nCreating visualization...")
    vis_image = visualize_results(image, h_splits, v_splits, grid_boxes, pred_classes, R, C)
    vis_image.save(args.output_image)
    print(f"  Saved to: {args.output_image}")

    # Create JSON output
    print(f"\nCreating JSON output...")
    json_output = create_json_output(
        h_splits, v_splits, grid_boxes, pred_classes, pred_probs,
        R, C, orig_width, orig_height
    )
    with open(args.output_json, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved to: {args.output_json}")

    print(f"\n{'='*60}")
    print(f"Inference completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

