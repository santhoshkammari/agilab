1. fixed groundtruth cells border and proper groundtruth
2.

results:

val loss
2025-10-10 11:54:15,007 - INFO -   Val Loss: 0.0264
2025-10-10 11:54:15,007 - INFO -   Val H - F1: 0.722, Precision: 0.751, Recall: 0.696
2025-10-10 11:54:15,007 - INFO -   Val V - F1: 0.811, Precision: 0.821, Recall: 0.802
2025-10-10 11:54:15,169 - INFO - 💾 Checkpoint saved: checkpoint_epoch_16.pth

train loss
📊 Epoch 16 Results:
2025-10-10 11:53:06,183 - INFO -   Loss: 0.0258
2025-10-10 11:53:06,183 - INFO -   H - F1: 0.743, Precision: 0.774, Recall: 0.716
2025-10-10 11:53:06,183 - INFO -   V - F1: 0.814, Precision: 0.825, Recall: 0.803

=============================================================
version 2

 forcing the model to learn from context rather than just pixel-perfect boundaries.


 ## **PURPOSE & REASONS**

### **What is this code doing?**

This code generates **ground truth labels** for training the **TABLET Split Model** (a table structure recognition system from the paper you uploaded).

**The workflow:**
1. **Input**: Table image + cell bounding boxes + OTSL structure tokens
2. **Output**: Two binary arrays (960 pixels each):
   - `horizontal_gt`: Which horizontal lines should be ROW splits (1 = split, 0 = not split)
   - `vertical_gt`: Which vertical lines should be COLUMN splits (1 = split, 0 = not split)

### **Why is this needed?**

According to the TABLET paper, the split model works as a **sequence labeling task**:
- It looks at each horizontal line and asks: "Is this a row separator?"
- It looks at each vertical line and asks: "Is this a column separator?"
- The model is trained using these ground truth arrays to learn patterns

---

## **YOUR ORIGINAL APPROACH**

```python
# Fixed 5-pixel width approach
split_width = 5  # Every split is exactly 5 pixels wide

# Mark split at position y with 5-pixel width
for offset in range(split_width):
    pos = y_int + offset
    horizontal_gt[pos] = 1
```

**Problems:**
1. ❌ **All gaps are treated equally** - A 2-pixel gap and a 50-pixel gap both get marked as 5 pixels
2. ❌ **Loses information** - Actual whitespace width is ignored
3. ❌ **Arbitrary choice** - Why 5? Why not 3 or 7?
4. ❌ **Bad for dense tables** - Financial tables (paper's focus) have very tight spacing

---

## **MY IMPROVED APPROACH**

### **What changed:**

```python
# DYNAMIC gap filling - mark the ENTIRE actual gap

# Between rows:
gap_start = row[i].bottom     # Where row i ends
gap_end = row[i+1].top        # Where row i+1 begins
mark_range(horizontal_gt, gap_start, gap_end)  # Mark ALL pixels in between
```

### **Key improvements:**

| Aspect | Old (Fixed) | New (Dynamic) |
|--------|-------------|---------------|
| **Gap width** | Always 5 pixels | Actual whitespace width (2px, 10px, 50px, etc.) |
| **Table boundaries** | Not marked | Marks space from image edge to table edge |
| **Dense tables** | Poor representation | Accurate - captures tight 2-3px gaps |
| **Training signal** | Artificial/arbitrary | Reflects real document structure |

---

## **WHY THESE IMPROVEMENTS MATTER**

### **1. Realistic Training Data**
```
Original approach (5px fixed):
|====CELL====|█████|====CELL====|  ← Always 5 pixels
               ↑
            Fixed width

New approach (dynamic):
|====CELL====|██|====CELL====|  ← Actual 2px gap (tight spacing)
|====CELL====|██████████|====CELL====|  ← Actual 10px gap
               ↑                ↑
          Real whitespace    Real whitespace
```

### **2. Better for Dense Financial Tables**

From the paper (page 3):
> "We adopt ResNet + FPN to provide a more concise yet **higher-resolution feature map**, making it better suited for the **large, densely populated tables with small fonts** commonly found in financial documents"

Dense tables = small gaps between rows/columns. Your dynamic approach captures these precisely.

### **3. Model learns actual patterns**

**With fixed width:**
- Model learns: "Look for arbitrary 5-pixel regions"
- Reality: Some gaps are 2px, some are 20px

**With dynamic width:**
- Model learns: "Gaps can be 2-50px, mark all of them"
- Reality: This matches actual documents

### **4. Handles variable spacing naturally**

```
Table with mixed spacing:
Header row:    20px gap below (lots of whitespace)
Data rows:     2px gaps (tightly packed)
Footer row:    15px gap above

Fixed approach: All marked as 5px (loses this information)
Dynamic approach: 20px, 2px, 2px, 2px, 15px (preserves structure)
```

---

## **VISUAL COMPARISON**

**Your screenshot shows** the red/green lines are now THICKER in some areas, THINNER in others - that's the dynamic gaps in action!

```
Before (fixed 5px):
|  CELL  |█████|  CELL  |█████|  CELL  |
          ↑ 5px        ↑ 5px

After (dynamic):
|  CELL  |██|  CELL  |████████|  CELL  |
          ↑ 2px       ↑ 8px (actual gaps)
```

---

## **ALIGNMENT WITH PAPER'S GOALS**

From page 6-7 of the paper:
> "We observe that **large, densely packed tables** are prevalent in financial announcements... Therefore, we consider it crucial to **maintain the resolution** of the feature map"

Your improvement:
✅ Maintains information about actual gap widths (preserves resolution)
✅ Better handles densely packed tables (2-3px gaps marked correctly)
✅ Provides richer training signal (model sees real whitespace patterns)

---

## **SUMMARY**

| Question | Answer |
|----------|--------|
| **Purpose** | Generate ground truth for training the TABLET split model to recognize row/column separators |
| **Original approach** | Fixed 5-pixel width for all splits |
| **Why original was problematic** | Ignores actual whitespace, bad for dense tables, arbitrary choice |
| **My improvement** | Dynamic gap filling - mark entire actual whitespace between rows/columns |
| **Why it's better** | Realistic, preserves table structure info, handles variable spacing, aligns with paper's focus on dense financial tables |

The improvement makes your training data **more accurate** and **more informative**, which should lead to better model performance on real-world dense tables! 🎯



====================================================
v3

1. subtables inside the table
2. wikipedia tables
3. sudoku mapped otsl 
4. llm generated contextually mappable otsl cells.