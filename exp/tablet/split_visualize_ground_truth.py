"""
Visualize ground truth split lines to verify correctness
"""
from datasets import load_dataset
from final_model import get_ground_truth
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def main():
    # Load dataset
    ds = load_dataset("ds4sd/FinTabNet_OTSL")

    idx = 3
    item = ds['train'][idx]

    print(f"Image {idx} info:")
    print(f"  Original size: {item['image'].size}")
    print(f"  Number of cells: {len(item['cells'][0])}")
    print(f"  OTSL: {item['otsl']}")

    # Get ground truth
    h_gt, v_gt = get_ground_truth(
        item['image'],
        item['html_restored'],
        item['cells'],
        item['otsl']
    )

    # Get original image and resize to 960x960
    img = item['image'].convert('RGB')
    img_resized = img.resize((960, 960))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Original image
    axes[0].imshow(img)
    axes[0].set_title(f'Original Image ({img.size[0]}x{img.size[1]})')
    axes[0].axis('off')

    # Plot 2: Resized with ground truth splits
    img_with_gt = img_resized.copy()
    draw = ImageDraw.Draw(img_with_gt)

    # Draw horizontal splits (rows) in red
    h_split_positions = [i for i, val in enumerate(h_gt) if val == 1]
    for y in h_split_positions:
        draw.line([(0, y), (960, y)], fill='red', width=2)

    # Draw vertical splits (columns) in blue
    v_split_positions = [i for i, val in enumerate(v_gt) if val == 1]
    for x in v_split_positions:
        draw.line([(x, 0), (x, 960)], fill='blue', width=2)

    axes[1].imshow(img_with_gt)
    axes[1].set_title(f'Ground Truth Splits\nH: {len(h_split_positions)} pixels, V: {len(v_split_positions)} pixels')
    axes[1].axis('off')

    # Plot 3: Show cell bounding boxes from original annotations
    img_with_bbox = img.copy()
    draw_bbox = ImageDraw.Draw(img_with_bbox)

    cells = item['cells'][0]
    print(f"\nFirst cell bbox: {cells[0]['bbox']}")
    print(f"Bbox length: {len(cells[0]['bbox'])}")

    for cell in cells:
        bbox = cell['bbox'][:4]  # Take only first 4 elements
        draw_bbox.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='green', width=2)

    axes[2].imshow(img_with_bbox)
    axes[2].set_title(f'Cell Bounding Boxes\n{len(cells)} cells')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'groundtruth_debug_{idx}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: groundtruth_debug_{idx}.png")

    # Print detailed split info
    print(f"\nHorizontal splits (rows):")
    print(f"  Total pixels marked: {sum(h_gt)}")
    print(f"  Unique positions: {len(set(h_split_positions))}")
    if h_split_positions:
        # Find continuous ranges
        ranges = []
        start = h_split_positions[0]
        prev = h_split_positions[0]
        for pos in h_split_positions[1:]:
            if pos != prev + 1:
                ranges.append((start, prev))
                start = pos
            prev = pos
        ranges.append((start, prev))
        print(f"  Split regions (y-axis): {ranges}")

    print(f"\nVertical splits (columns):")
    print(f"  Total pixels marked: {sum(v_gt)}")
    print(f"  Unique positions: {len(set(v_split_positions))}")
    if v_split_positions:
        # Find continuous ranges
        ranges = []
        start = v_split_positions[0]
        prev = v_split_positions[0]
        for pos in v_split_positions[1:]:
            if pos != prev + 1:
                ranges.append((start, prev))
                start = pos
            prev = pos
        ranges.append((start, prev))
        print(f"  Split regions (x-axis): {ranges}")

if __name__ == "__main__":
    main()

