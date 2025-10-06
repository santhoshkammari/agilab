def vis_otsl_with_border(sample):
    cells_flat = sample['cells'][0]
    otsl = sample['otsl']
    img = sample['image']
    # parse otsl
    grid = []
    current_row = []
    cell_idx = 0
    for token in otsl:
        if token == 'nl':
            if current_row:
                grid.append(current_row)
                current_row = []
        elif token == 'fcel' or token == 'ecel':
            current_row.append({'type': token, 'cell_idx': cell_idx})
            cell_idx += 1
        elif token in ['lcel', 'ucel', 'xcel']:
            current_row.append({'type': token, 'cell_idx': None})
    if current_row:
        grid.append(current_row)
    
    # find table boundary - min/max of all cell bboxes
    all_x1 = [c['bbox'][0] for c in cells_flat]
    all_y1 = [c['bbox'][1] for c in cells_flat]
    all_x2 = [c['bbox'][2] for c in cells_flat]
    all_y2 = [c['bbox'][3] for c in cells_flat]
    table_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]
    
    # derive splits
    row_splits = []
    for row in grid:
        row_cell_indices = [item['cell_idx'] for item in row if item['cell_idx'] is not None]
        if row_cell_indices:
            max_y = max([cells_flat[i]['bbox'][3] for i in row_cell_indices])
            row_splits.append(max_y)
    
    num_cols = len(grid[0])
    col_splits = []
    for col_idx in range(num_cols):
        col_max_x = []
        for row in grid:
            if col_idx < len(row) and row[col_idx]['cell_idx'] is not None:
                # Only use this cell if the next column is NOT 'lcel' (not a horizontal merge)
                next_is_lcel = (col_idx + 1 < len(row) and row[col_idx + 1]['type'] == 'lcel')
                if not next_is_lcel:
                    cell_id = row[col_idx]['cell_idx']
                    col_max_x.append(cells_flat[cell_id]['bbox'][2])
        if col_max_x:
            col_splits.append(max(col_max_x))
    
    # visualize
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    # draw table border
    draw.rectangle(table_bbox, outline='yellow', width=2)
    # draw internal splits
    for y in row_splits:
        draw.line([(table_bbox[0], y), (table_bbox[2], y)], fill='red', width=2)
    for x in col_splits:
        draw.line([(x, table_bbox[1]), (x, table_bbox[3])], fill='green', width=2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_copy)
    plt.title(f'yellow=border, red=rows, green=cols | {len(row_splits)}×{len(col_splits)} grid')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

vis_otsl_with_border(ds['train'][10])
