"""
Visualize predictions from trained model on local images folder
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from split_model import SplitModel
from PIL import Image, ImageDraw
import numpy as np
import glob
import os

class LocalImageDataset(Dataset):
    """Dataset for loading images from a local folder"""
    def __init__(self, image_folder):
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")) +
                                  glob.glob(os.path.join(image_folder, "*.jpg")) +
                                  glob.glob(os.path.join(image_folder, "*.jpeg")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")

        self.transform = transforms.Compose([
            transforms.Resize((960, 960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Found {len(self.image_paths)} images in {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_transformed = self.transform(image)
        return image_transformed, image, image_path

def get_middle_of_groups(binary_array):
    """
    Find groups of consecutive 1's and return only the middle index of each group.
    Example: [0,0,1,1,1,1,1,0,0,1,1,0] -> [0,0,0,0,1,0,0,0,1,0,0]
    """
    result = np.zeros_like(binary_array)
    i = 0
    n = len(binary_array)

    while i < n:
        if binary_array[i] == 1:
            # Found start of a group
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            end = i - 1

            # Get middle index
            middle = (start + end) // 2
            result[middle] = 1
        else:
            i += 1

    return result

def visualize_prediction(model, dataset, idx, device, output_folder, threshold=0.5):
    """Visualize prediction for a single image (no ground truth)"""
    model.eval()

    # Get sample
    image_tensor, original_image, image_path = dataset[idx]

    # Predict
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        h_pred, v_pred = model(image_batch)  # [1, 480]

        # Upsample to 960 for visualization
        h_pred = h_pred.repeat_interleave(2, dim=1)  # [1, 960]
        v_pred = v_pred.repeat_interleave(2, dim=1)  # [1, 960]

        h_pred = h_pred.squeeze(0).cpu()
        v_pred = v_pred.squeeze(0).cpu()

    # Apply threshold
    h_binary = (h_pred > threshold).float().numpy()
    v_binary = (v_pred > threshold).float().numpy()

    # Get only middle of grouped 1's for cleaner visualization
    h_binary_clean = get_middle_of_groups(h_binary)
    v_binary_clean = get_middle_of_groups(v_binary)

    # Count predictions (use cleaned version)
    h_splits = h_binary_clean.sum()
    v_splits = v_binary_clean.sum()
    pred_rows = int(h_splits) + 1
    pred_cols = int(v_splits) + 1

    print(f"\nImage {idx}: {os.path.basename(image_path)}")
    print(f"  H Splits: {h_splits:.0f} | Pred Rows: {pred_rows}")
    print(f"  V Splits: {v_splits:.0f} | Pred Cols: {pred_cols}")
    print(f"  Total Cells: {pred_rows * pred_cols}")
    print(f"  H pred range: [{h_pred.min():.3f}, {h_pred.max():.3f}]")
    print(f"  V pred range: [{v_pred.min():.3f}, {v_pred.max():.3f}]")

    # Visualize
    W, H = original_image.size

    # Zoom factor for larger images
    zoom_factor = 1.5
    W_zoomed = int(W * zoom_factor)
    H_zoomed = int(H * zoom_factor)

    # Resize images for better visibility
    original_zoomed = original_image.resize((W_zoomed, H_zoomed), Image.LANCZOS)

    # Info panel dimensions
    info_height = 150
    label_height = 60
    padding = 40

    try:
        from PIL import ImageFont
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        font_title = None
        font_text = None
        font_label = None

    # Create visualization with 3 images stacked vertically
    total_width = W_zoomed + padding * 2
    total_height = info_height + (label_height + H_zoomed) * 3 + padding * 5
    vis_image = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(vis_image)

    # Info Panel at top
    info_x = padding
    info_y = padding

    draw.rectangle([info_x, info_y, total_width - padding, info_y + info_height],
                   fill='#e3f2fd', outline='#1565c0', width=4)

    draw.text((info_x + 20, info_y + 20), "PREDICTED SPLITS", fill='#0d47a1', font=font_title)
    draw.text((info_x + 20, info_y + 70), f"Rows: {pred_rows}", fill='black', font=font_text)
    draw.text((info_x + 20, info_y + 105), f"Columns: {pred_cols}", fill='black', font=font_text)
    draw.text((info_x + 300, info_y + 70), f"H Splits: {int(h_splits)}", fill='#c62828', font=font_text)
    draw.text((info_x + 300, info_y + 105), f"V Splits: {int(v_splits)}", fill='#1565c0', font=font_text)

    # 1. Original image (top)
    y_pos = info_height + padding * 2
    draw.rectangle([padding, y_pos, total_width - padding, y_pos + label_height],
                   fill='#f5f5f5', outline='#666666', width=2)
    draw.text((padding + 20, y_pos + 15), "Original Image", fill='#333333', font=font_label)

    vis_image.paste(original_zoomed, (padding, y_pos + label_height))

    # 2. Raw predictions (middle) - with all thick lines
    y_pos = info_height + padding * 3 + label_height + H_zoomed
    draw.rectangle([padding, y_pos, total_width - padding, y_pos + label_height],
                   fill='#fff3e0', outline='#ff9800', width=2)
    draw.text((padding + 20, y_pos + 15), "Raw Model Predictions (All 1's)", fill='#e65100', font=font_label)

    # Create raw prediction image with all 1's (before cleaning)
    raw_pred_image = original_image.copy()
    draw_raw = ImageDraw.Draw(raw_pred_image)

    # Draw ALL predicted horizontal lines (red) - raw, not cleaned
    for y in range(960):
        if h_binary[y] == 1:
            y_scaled = int(y * H / 960)
            draw_raw.line([(0, y_scaled), (W, y_scaled)], fill='#ff0000', width=2)

    # Draw ALL predicted vertical lines (blue) - raw, not cleaned
    for x in range(960):
        if v_binary[x] == 1:
            x_scaled = int(x * W / 960)
            draw_raw.line([(x_scaled, 0), (x_scaled, H)], fill='#0000ff', width=2)

    # Zoom raw prediction image
    raw_pred_zoomed = raw_pred_image.resize((W_zoomed, H_zoomed), Image.LANCZOS)
    vis_image.paste(raw_pred_zoomed, (padding, y_pos + label_height))

    # 3. Cleaned predictions (bottom) - only middle lines
    y_pos = info_height + padding * 4 + (label_height + H_zoomed) * 2
    draw.rectangle([padding, y_pos, total_width - padding, y_pos + label_height],
                   fill='#e3f2fd', outline='#1565c0', width=2)
    draw.text((padding + 20, y_pos + 15), "Cleaned Predictions (Middle Only)", fill='#0d47a1', font=font_label)

    # Create cleaned prediction image
    pred_image = original_image.copy()
    draw_pred = ImageDraw.Draw(pred_image)

    # Draw predicted horizontal lines (red) - using cleaned version
    for y in range(960):
        if h_binary_clean[y] == 1:
            y_scaled = int(y * H / 960)
            draw_pred.line([(0, y_scaled), (W, y_scaled)], fill='#ff0000', width=3)

    # Draw predicted vertical lines (blue) - using cleaned version
    for x in range(960):
        if v_binary_clean[x] == 1:
            x_scaled = int(x * W / 960)
            draw_pred.line([(x_scaled, 0), (x_scaled, H)], fill='#0000ff', width=3)

    # Zoom prediction image
    pred_zoomed = pred_image.resize((W_zoomed, H_zoomed), Image.LANCZOS)
    vis_image.paste(pred_zoomed, (padding, y_pos + label_height))

    # Save to output folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f'prediction_{base_name}.png')
    vis_image.save(output_path)
    print(f"  Saved: {output_path}")

    return pred_rows, pred_cols

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize table split predictions on local images')
    parser.add_argument('--image-folder', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--output-folder', type=str, default='predictions_output',
                        help='Path to folder for saving predictions (default: predictions_output)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint (if not specified, will search common locations)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary predictions (default: 0.5)')
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Number of images to process (-1 for all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder: {args.output_folder}")

    # Load dataset from local folder
    print(f"\nLoading images from: {args.image_folder}")
    dataset = LocalImageDataset(args.image_folder)

    # Load model
    print("\nLoading model...")
    model = SplitModel().to(device)

    # Try to find the best model
    if args.model_path:
        checkpoint_path = args.model_path
    else:
        possible_paths = [
            'best_split_model.pth',
            '/home/ng6309/datascience/santhosh/experiments/tablet/best_split_model.pth',
            '/home/ng6309/datascience/santhosh/experiments/tablet/runs/tablet_split_20251006_214335/best_split_model.pth',
        ]

        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("ERROR: No trained model found! Please specify --model-path or ensure model exists in default locations")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nModel trained for {checkpoint['epoch']} epochs")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    if 'val_h_f1' in checkpoint:
        print(f"Val H F1: {checkpoint['val_h_f1']:.3f}")
        print(f"Val V F1: {checkpoint['val_v_f1']:.3f}")

    # Determine number of images to process
    num_samples = len(dataset) if args.num_images == -1 else min(args.num_images, len(dataset))

    # Visualize images
    print("\n" + "="*60)
    print(f"Visualizing predictions for {num_samples} images")
    print("="*60)

    for idx in range(num_samples):
        pred_rows, pred_cols = visualize_prediction(model, dataset, idx, device, args.output_folder, threshold=args.threshold)

    print("\n" + "="*60)
    print(f"Completed processing {num_samples} images")
    print(f"All predictions saved to: {args.output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()
