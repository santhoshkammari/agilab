# NotebookEdit Tool - Comprehensive Documentation

## Overview
The NotebookEdit tool modifies Jupyter notebook (.ipynb) cells with support for replacing, inserting, and deleting cells. It maintains notebook integrity and handles cell operations while preserving notebook structure.

## Function Signature
```json
{
  "name": "NotebookEdit",
  "parameters": {
    "notebook_path": "string (required)",
    "new_source": "string (required)",
    "cell_id": "string (optional)",
    "cell_type": "enum (optional)",
    "edit_mode": "enum (optional)"
  }
}
```

## Parameters

### notebook_path (required)
- **Type**: string
- **Purpose**: Absolute path to Jupyter notebook file
- **Constraint**: Must be absolute path (not relative)
- **Format**: `/absolute/path/to/notebook.ipynb`
- **Validation**: File must exist and be valid .ipynb format

### new_source (required)
- **Type**: string
- **Purpose**: New content for the cell
- **Format**: Raw cell content (code, markdown, or raw text)
- **Encoding**: UTF-8 text content
- **Multiline**: Supports multiline content with proper newlines

### cell_id (optional)
- **Type**: string
- **Purpose**: ID of cell to edit (replace/delete) or reference for insert position
- **Insert Behavior**: New cell inserted after this cell ID
- **Default**: Operations target beginning of notebook if not specified
- **Format**: Cell UUID from notebook metadata

### cell_type (optional)
- **Type**: enum
- **Values**: `"code"` | `"markdown"`
- **Purpose**: Type of cell to create or modify
- **Default**: Inherits current cell type for replace mode
- **Required**: Must specify for insert mode

### edit_mode (optional)
- **Type**: enum
- **Values**: `"replace"` | `"insert"` | `"delete"`
- **Default**: `"replace"`
- **Behavior**:
  - `replace`: Replace content of existing cell
  - `insert`: Add new cell at specified position
  - `delete`: Remove existing cell

## Edit Modes

### Replace Mode (default)
```json
{
  "notebook_path": "/analysis/data.ipynb",
  "cell_id": "cell-123",
  "new_source": "import pandas as pd\nimport numpy as np\n\n# Updated imports with better organization",
  "edit_mode": "replace"
}
```
- **Behavior**: Replaces content of existing cell
- **Preservation**: Maintains cell metadata and position
- **Type**: Cell type can be changed if specified

### Insert Mode
```json
{
  "notebook_path": "/analysis/data.ipynb", 
  "cell_id": "cell-123",
  "new_source": "# New Analysis Section\n\nThis section performs additional analysis.",
  "cell_type": "markdown",
  "edit_mode": "insert"
}
```
- **Behavior**: Creates new cell after specified cell_id
- **Position**: If cell_id not specified, inserts at beginning
- **Requirement**: cell_type must be specified
- **Indexing**: All subsequent cells shift down

### Delete Mode
```json
{
  "notebook_path": "/analysis/data.ipynb",
  "cell_id": "cell-123", 
  "new_source": "",
  "edit_mode": "delete"
}
```
- **Behavior**: Removes specified cell completely
- **Source**: new_source parameter ignored but still required
- **Indexing**: All subsequent cells shift up
- **Permanent**: Cell and all its outputs are lost

## Cell Types

### Code Cells
```python
# Python code cell content
{
  "new_source": "import matplotlib.pyplot as plt\n\n# Create visualization\nfig, ax = plt.subplots()\nax.plot([1, 2, 3, 4], [1, 4, 2, 3])\nplt.title('Sample Plot')\nplt.show()",
  "cell_type": "code"
}

# R code cell (if R kernel)
{
  "new_source": "library(ggplot2)\n\n# Create plot\nggplot(mtcars, aes(x=wt, y=mpg)) + \n  geom_point() +\n  labs(title='Car Weight vs MPG')",
  "cell_type": "code"
}
```

### Markdown Cells
```markdown
# Documentation and explanations
{
  "new_source": "# Data Analysis Report\n\n## Overview\n\nThis notebook analyzes sales data from Q4 2023.\n\n### Key Findings\n\n- Sales increased by 15%\n- Customer retention improved\n- New market segments identified\n\n### Methodology\n\n1. Data cleaning and preprocessing\n2. Statistical analysis\n3. Visualization and reporting",
  "cell_type": "markdown"
}
```

## Common Operations

### Code Cell Updates
```json
{
  "notebook_path": "/projects/ml-model.ipynb",
  "cell_id": "training-cell",
  "new_source": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score\n\n# Updated model with better parameters\nmodel = RandomForestClassifier(\n    n_estimators=200,\n    max_depth=10,\n    random_state=42\n)\n\n# Cross-validation\nscores = cross_val_score(model, X_train, y_train, cv=5)\nprint(f'CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')",
  "edit_mode": "replace"
}
```

### Documentation Enhancement
```json
{
  "notebook_path": "/research/experiment.ipynb",
  "cell_id": "intro-cell",
  "new_source": "# Experiment: Neural Network Architecture Comparison\n\n## Hypothesis\n\nWe hypothesize that deeper networks with residual connections will outperform shallow networks on our image classification task.\n\n## Experimental Setup\n\n- **Dataset**: CIFAR-10 (60,000 32x32 color images)\n- **Models**: ResNet-18, ResNet-34, Simple CNN\n- **Metrics**: Accuracy, F1-score, Training time\n- **Hardware**: GPU Tesla V100\n\n## Expected Outcomes\n\nResNet architectures should achieve >90% accuracy while simple CNN should achieve ~85%.",
  "cell_type": "markdown",
  "edit_mode": "replace"
}
```

### Cell Insertion
```json
{
  "notebook_path": "/analysis/sales.ipynb",
  "cell_id": "data-loading",
  "new_source": "# Data Quality Check\n\n# Check for missing values\nprint(\"Missing values per column:\")\nprint(df.isnull().sum())\n\n# Check data types\nprint(\"\\nData types:\")\nprint(df.dtypes)\n\n# Basic statistics\nprint(\"\\nBasic statistics:\")\nprint(df.describe())",
  "cell_type": "code",
  "edit_mode": "insert"
}
```

### Cell Removal
```json
{
  "notebook_path": "/debug/analysis.ipynb",
  "cell_id": "debug-cell-temp",
  "new_source": "",
  "edit_mode": "delete"
}
```

## Advanced Usage Patterns

### Refactoring Code Cells
```json
{
  "notebook_path": "/ml/feature-engineering.ipynb",
  "cell_id": "feature-creation",
  "new_source": "def create_features(df):\n    \"\"\"Create engineered features for machine learning.\"\"\"\n    \n    # Date features\n    df['year'] = df['date'].dt.year\n    df['month'] = df['date'].dt.month\n    df['day_of_week'] = df['date'].dt.dayofweek\n    \n    # Interaction features\n    df['price_per_unit'] = df['total_price'] / df['quantity']\n    df['customer_value_segment'] = pd.cut(\n        df['customer_lifetime_value'], \n        bins=[0, 100, 500, 1000, float('inf')],\n        labels=['Low', 'Medium', 'High', 'Premium']\n    )\n    \n    return df\n\n# Apply feature engineering\ndf_features = create_features(df.copy())\nprint(f\"Created {df_features.shape[1] - df.shape[1]} new features\")",
  "edit_mode": "replace"
}
```

### Adding Analysis Sections
```json
{
  "notebook_path": "/research/analysis.ipynb",
  "cell_id": "preliminary-results",
  "new_source": "## Statistical Significance Testing\n\nBefore drawing conclusions, we need to test the statistical significance of our findings.",
  "cell_type": "markdown",
  "edit_mode": "insert"
}
```

### Error Correction
```json
{
  "notebook_path": "/projects/data-pipeline.ipynb",
  "cell_id": "buggy-code",
  "new_source": "# Fixed version with proper error handling\ntry:\n    processed_data = []\n    for item in raw_data:\n        if item is not None and 'value' in item:\n            processed_item = {\n                'id': item.get('id', 'unknown'),\n                'value': float(item['value']),\n                'timestamp': pd.to_datetime(item.get('timestamp'))\n            }\n            processed_data.append(processed_item)\n        else:\n            print(f\"Skipping invalid item: {item}\")\n    \n    df_processed = pd.DataFrame(processed_data)\n    print(f\"Successfully processed {len(df_processed)} items\")\n    \nexcept Exception as e:\n    print(f\"Error processing data: {e}\")\n    df_processed = pd.DataFrame()  # Empty fallback",
  "edit_mode": "replace"
}
```

## Integration Patterns

### Development Workflow
```
1. NotebookRead(notebook_path="/project/analysis.ipynb")
2. Identify cells needing updates
3. NotebookEdit(notebook_path="/project/analysis.ipynb", ...)  // Update cells
4. Bash(command="jupyter nbconvert --execute analysis.ipynb")  // Test execution
```

### Code Review Process
```
1. NotebookRead(notebook_path="/review/submission.ipynb")
2. Analyze code quality and logic
3. NotebookEdit(notebook_path="/review/submission.ipynb", ...)  // Add feedback
4. NotebookEdit(edit_mode="insert", ...)  // Add review comments
```

### Template Application
```
1. Write(file_path="/templates/analysis-template.ipynb", content="...")
2. NotebookEdit(notebook_path="/templates/analysis-template.ipynb", ...)  // Customize
3. Multiple NotebookEdit calls to populate template sections
```

## Error Handling

### Common Errors

#### Notebook Not Found
```json
{
  "error": "Notebook file not found: /path/to/notebook.ipynb",
  "code": "FILE_NOT_FOUND"
}
```

#### Invalid Cell ID
```json
{
  "error": "Cell ID not found: invalid-cell-id",
  "available_cells": ["cell-abc", "cell-def", "cell-ghi"],
  "suggestion": "Use NotebookRead to get valid cell IDs"
}
```

#### Missing Cell Type for Insert
```json
{
  "error": "cell_type required for insert mode",
  "details": "Specify 'code' or 'markdown' when inserting new cells"
}
```

#### Invalid Notebook Format
```json
{
  "error": "Invalid notebook format after edit",
  "details": "Notebook structure corrupted, operation rolled back"
}
```

### Recovery Strategies
1. **Read First**: Use NotebookRead to understand current structure
2. **Validate IDs**: Check available cell IDs before editing
3. **Backup**: Keep notebook under version control
4. **Test Execution**: Run notebook after major changes

## Best Practices

### Safe Editing
1. **Read Before Edit**: Always use NotebookRead first
2. **Incremental Changes**: Make small, focused edits
3. **Test Execution**: Verify notebook runs after changes
4. **Version Control**: Keep notebooks in git for rollback

### Content Guidelines
```python
# ✅ Good - Clear, executable code
{
  "new_source": "# Load required libraries\nimport pandas as pd\nimport numpy as np\n\n# Load data with error handling\ntry:\n    df = pd.read_csv('data.csv')\n    print(f\"Loaded {len(df)} rows\")\nexcept FileNotFoundError:\n    print(\"Data file not found\")\n    df = pd.DataFrame()"
}

# ❌ Poor - Unclear, potentially broken code
{
  "new_source": "df=pd.read_csv('data.csv')\nprint(df)"
}
```

### Documentation Standards
```markdown
# ✅ Good - Comprehensive documentation
{
  "new_source": "# Data Preprocessing\n\n## Overview\n\nThis section cleans and prepares the raw data for analysis.\n\n## Steps\n\n1. Remove duplicates\n2. Handle missing values\n3. Normalize numerical features\n4. Encode categorical variables\n\n## Output\n\nClean dataset ready for machine learning",
  "cell_type": "markdown"
}
```

## Performance Considerations

### Notebook Size Impact
- **Small Notebooks** (<1MB): Fast editing operations
- **Medium Notebooks** (1MB-10MB): Good performance
- **Large Notebooks** (>10MB): May take longer, especially with many cells
- **Cell Count**: Operations slower with hundreds of cells

### Memory Management
- **Cell Content**: Large cells consume more memory
- **Output Preservation**: Cell outputs maintained during edits
- **JSON Processing**: Notebook parsed and reconstructed

### Optimization Strategies
1. **Batch Edits**: Group related changes when possible
2. **Targeted Operations**: Use specific cell IDs rather than bulk operations
3. **Clean Outputs**: Clear large outputs before editing if not needed
4. **File Size Management**: Keep notebooks focused and manageable

## Security Considerations

### Content Validation
- **Code Safety**: Review code cells for potentially harmful operations
- **Data Privacy**: Check for sensitive data in cell content
- **Execution Risk**: Remember code cells can execute arbitrary code
- **Output Sensitivity**: Consider if cell outputs contain sensitive information

### File Safety
- **Backup Strategy**: Maintain backups of important notebooks
- **Permission Handling**: Respect file system permissions
- **Path Validation**: Ensure notebook paths are safe and authorized
- **Version Control**: Track changes for audit and rollback

## Tool Comparison

### NotebookEdit vs Edit/MultiEdit
- **NotebookEdit**: Jupyter-specific cell operations
- **Edit/MultiEdit**: Generic text file editing

### NotebookEdit vs Write
- **NotebookEdit**: Preserves notebook structure and metadata
- **Write**: Would overwrite entire notebook file

### NotebookEdit vs NotebookRead
- **NotebookEdit**: Modifies notebook content
- **NotebookRead**: Read-only notebook inspection

## Advanced Features

### Cell Metadata Preservation
```python
# Cell metadata is automatically preserved
{
  "metadata": {
    "tags": ["analysis", "important"],
    "collapsed": false,
    "scrolled": true
  }
}
```

### Kernel Information Maintenance
```python
# Notebook kernel info preserved
{
  "kernelspec": {
    "display_name": "Python 3",
    "language": "python", 
    "name": "python3"
  }
}
```

### Output Handling
- **Code Cell Edits**: Outputs cleared when source changes
- **Markdown Cells**: No outputs to manage
- **Execution Counts**: Reset for modified code cells