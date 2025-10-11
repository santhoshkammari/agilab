# Evaluation Report

## Model Performance Summary

| Dataset Split | Dataset Size | Epoch | Test Loss | Avg F1 | H F1 | H Precision | H Recall | H Count Error | V F1 | V Precision | V Recall | V Count Error |
|--------------|--------------|-------|-----------|---------|------|-------------|----------|---------------|------|-------------|----------|---------------|
| Test (original) | 10,397 | 15 | 0.0566 | 0.935 | 0.911 | 0.915 | 0.909 | 214.34 | 0.958 | 0.949 | 0.970 | 321.32 |
| Val (PubTabNet_OTSL_full) | 6,942 | 15 | 0.1409 | 0.817 | 0.710 | 0.654 | 0.779 | 1577.85 | 0.923 | 0.916 | 0.931 | 273.69 |

## Model Details

- **Model Parameters**: 10,668,196
- **Model Checkpoint**: best_split_model.pth
- **Dataset Path**: /home/ng6309/datascience/santhosh/datasets/PubTabNet_OTSL_full

## Key Observations

1. **Original Test Set**: Higher performance with Avg F1 of 0.935
2. **PubTabNet Val Set**: Lower performance with Avg F1 of 0.817
3. **Horizontal Splits (H)**: Significant performance drop on PubTabNet (0.911 → 0.710 F1)
4. **Vertical Splits (V)**: More consistent performance across datasets (0.958 → 0.923 F1)
5. **Count Errors**: H count error increased significantly on PubTabNet (214.34 → 1577.85)

---
*Generated: 2025-10-12*
