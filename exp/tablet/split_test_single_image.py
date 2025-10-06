"""
Test single gradient update on one image
"""
import torch
from datasets import load_dataset
from final_model import SplitModel, TableDataset, focal_loss
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, idx, device, title="Predictions"):
    """Visualize model predictions"""
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

    # Binary predictions
    h_binary = (h_pred_np > 0.5).astype(int)
    v_binary = (v_pred_np > 0.5).astype(int)

    # Draw predictions
    for y in range(960):
        if h_binary[y] == 1:
            draw.line([(0, y), (960, y)], fill='red', width=2)

    for x in range(960):
        if v_binary[x] == 1:
            draw.line([(x, 0), (x, 960)], fill='blue', width=2)

    # Calculate accuracy
    h_gt_np = h_gt.numpy()
    v_gt_np = v_gt.numpy()
    h_acc = (h_binary == h_gt_np).sum() / 960
    v_acc = (v_binary == v_gt_np).sum() / 960

    print(f"\n{title}:")
    print(f"  H: GT={h_gt_np.sum():.0f}, Pred={h_binary.sum():.0f}, Acc={h_acc:.3f}")
    print(f"  V: GT={v_gt_np.sum():.0f}, Pred={v_binary.sum():.0f}, Acc={v_acc:.3f}")

    return img_resized, h_acc, v_acc

def main():
    print("=" * 80)
    print("Single Image Gradient Update Test")
    print("=" * 80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    ds = load_dataset("ds4sd/FinTabNet_OTSL")
    dataset = TableDataset(ds['train'])
    idx = 3

    # Get single sample
    img_tensor, h_gt, v_gt = dataset[idx]
    img_batch = img_tensor.unsqueeze(0).to(device)
    h_gt_batch = h_gt.unsqueeze(0).to(device)
    v_gt_batch = v_gt.unsqueeze(0).to(device)

    print(f"\nImage {idx}:")
    print(f"  Image shape: {img_batch.shape}")
    print(f"  H GT shape: {h_gt_batch.shape}")
    print(f"  V GT shape: {v_gt_batch.shape}")

    # Initialize model
    model = SplitModel().to(device)

    # BEFORE training
    print("\n" + "=" * 80)
    print("BEFORE TRAINING:")
    print("=" * 80)
    img_before, h_acc_before, v_acc_before = visualize_predictions(
        model, dataset, idx, device, "Before Training"
    )

    # Calculate initial loss
    model.train()
    h_pred, v_pred = model(img_batch)
    h_loss = focal_loss(h_pred, h_gt_batch, alpha=1.0, gamma=2.0)
    v_loss = focal_loss(v_pred, v_gt_batch, alpha=1.0, gamma=2.0)
    initial_loss = h_loss + v_loss
    print(f"\nInitial Loss: {initial_loss.item():.4f} (H: {h_loss.item():.4f}, V: {v_loss.item():.4f})")

    # Train on this ONE image multiple times
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    print("\n" + "=" * 80)
    print("TRAINING ON SINGLE IMAGE (100 steps):")
    print("=" * 80)

    for step in range(100):
        optimizer.zero_grad()

        h_pred, v_pred = model(img_batch)
        h_loss = focal_loss(h_pred, h_gt_batch, alpha=1.0, gamma=2.0)
        v_loss = focal_loss(v_pred, v_gt_batch, alpha=1.0, gamma=2.0)
        total_loss = h_loss + v_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: Loss={total_loss.item():.4f} (H: {h_loss.item():.4f}, V: {v_loss.item():.4f})")

    # AFTER training
    print("\n" + "=" * 80)
    print("AFTER TRAINING:")
    print("=" * 80)
    img_after, h_acc_after, v_acc_after = visualize_predictions(
        model, dataset, idx, device, "After Training"
    )

    # Final loss
    model.eval()
    with torch.no_grad():
        h_pred, v_pred = model(img_batch)
        h_loss = focal_loss(h_pred, h_gt_batch, alpha=1.0, gamma=2.0)
        v_loss = focal_loss(v_pred, v_gt_batch, alpha=1.0, gamma=2.0)
        final_loss = h_loss + v_loss

    print(f"\nFinal Loss: {final_loss.item():.4f} (H: {h_loss.item():.4f}, V: {v_loss.item():.4f})")
    print(f"Loss Reduction: {initial_loss.item() - final_loss.item():.4f}")

    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(img_before)
    axes[0].set_title(f'Before Training\nH Acc: {h_acc_before:.3f}, V Acc: {v_acc_before:.3f}\nLoss: {initial_loss.item():.4f}')
    axes[0].axis('off')

    axes[1].imshow(img_after)
    axes[1].set_title(f'After 100 Steps\nH Acc: {h_acc_after:.3f}, V Acc: {v_acc_after:.3f}\nLoss: {final_loss.item():.4f}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('single_image_training_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: single_image_training_test.png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Initial Loss: {initial_loss.item():.4f}")
    print(f"Final Loss:   {final_loss.item():.4f}")
    print(f"Improvement:  {initial_loss.item() - final_loss.item():.4f}")
    print(f"\nH Accuracy: {h_acc_before:.3f} → {h_acc_after:.3f} (Δ {h_acc_after - h_acc_before:.3f})")
    print(f"V Accuracy: {v_acc_before:.3f} → {v_acc_after:.3f} (Δ {v_acc_after - v_acc_before:.3f})")

    if final_loss < initial_loss:
        print("\n✅ Training is working! Loss decreased successfully.")
    else:
        print("\n❌ Warning: Loss did not decrease!")

    print("=" * 80)

if __name__ == "__main__":
    main()

