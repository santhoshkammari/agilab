"""
Visualize predictions with different thresholds
"""
import torch
from datasets import load_dataset
from final_model import SplitModel, TableDataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def visualize_with_threshold(model, dataset, idx, device, threshold):
    model.eval()

    img_tensor, h_gt, v_gt = dataset[idx]

    # Get original image
    original_item = dataset.hf_dataset[idx]
    img = original_item['image'].convert('RGB')
    img_resized = img.resize((960, 960))
    draw = ImageDraw.Draw(img_resized)

    # Run inference
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0).to(device)
        h_pred, v_pred = model(img_batch)
        h_pred_np = h_pred.squeeze(0).cpu().numpy()
        v_pred_np = v_pred.squeeze(0).cpu().numpy()

    # Binary predictions with custom threshold
    h_binary = (h_pred_np > threshold).astype(int)
    v_binary = (v_pred_np > threshold).astype(int)

    # Draw predictions
    for y in range(960):
        if h_binary[y] == 1:
            draw.line([(0, y), (960, y)], fill='yellow', width=2)

    for x in range(960):
        if v_binary[x] == 1:
            draw.line([(x, 0), (x, 960)], fill='yellow', width=2)

    # Calculate accuracy
    h_gt_np = h_gt.numpy()
    v_gt_np = v_gt.numpy()
    h_acc = (h_binary == h_gt_np).sum() / 960
    v_acc = (v_binary == v_gt_np).sum() / 960

    return img_resized, h_acc, v_acc, h_binary.sum(), v_binary.sum()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SplitModel().to(device)
    checkpoint = torch.load('best_model_100img.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    ds = load_dataset("ds4sd/FinTabNet_OTSL")
    dataset = TableDataset(ds['val'])

    # Test different thresholds on image 0
    idx = 0
    _, h_gt, v_gt = dataset[idx]

    print(f"Image {idx} Ground Truth: H={h_gt.sum():.0f}, V={v_gt.sum():.0f}")
    print("\nTesting different thresholds:")

    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = []

    for thresh in thresholds:
        img, h_acc, v_acc, h_pred, v_pred = visualize_with_threshold(
            model, dataset, idx, device, thresh
        )
        results.append((thresh, img, h_acc, v_acc, h_pred, v_pred))
        print(f"  Threshold {thresh}: H_pred={h_pred}, V_pred={v_pred}, "
              f"H_acc={h_acc:.3f}, V_acc={v_acc:.3f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i, (thresh, img, h_acc, v_acc, h_pred, v_pred) in enumerate(results):
        axes[i].imshow(img)
        title = (f"Threshold = {thresh}\n"
                f"H: Pred={h_pred}, Acc={h_acc:.3f}\n"
                f"V: Pred={v_pred}, Acc={v_acc:.3f}")
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: threshold_comparison.png")

if __name__ == "__main__":
    main()

