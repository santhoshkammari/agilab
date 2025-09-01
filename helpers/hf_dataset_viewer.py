#!/usr/bin/env python3
"""
HuggingFace Dataset Viewer

A command-line tool for viewing and inspecting HuggingFace datasets with flexible
display options, sample selection, and field truncation capabilities.

Features:
- Load datasets from local files or HuggingFace Hub
- Support for multiple file formats (JSON, JSONL, CSV, Parquet, TSV, TXT)
- Two display modes: normal (detailed panels) and table (compact grid)
- Flexible sample selection by index or feature value matching
- Field truncation with slice expressions for better readability
- Rich console output with syntax highlighting

Usage Examples:
    # View first sample of a dataset
    python hf_dataset_viewer.py dataset_name

    # View multiple samples in table format
    python hf_dataset_viewer.py dataset_name --samples 5 --table

    # View specific sample by index (0-based)
    python hf_dataset_viewer.py dataset_name --index 10

    # View sample where feature 'label' equals 'positive'
    python hf_dataset_viewer.py dataset_name --index positive --index-feature label

    # View all samples
    python hf_dataset_viewer.py dataset_name --samples -1

    # Truncate fields using slice expressions
    python hf_dataset_viewer.py dataset_name --exp "text[:100],query[2:50]"
    
    # Show only specific columns
    python hf_dataset_viewer.py dataset_name -c "query,sample"

Arguments:
    name                    Dataset path/name (local file/directory or HuggingFace dataset)
    --samples, -s          Number of samples to show (default: 1, -1 for all)
    --view, -v             Display format: 'normal' or 'table' (default: normal)
    --table, -t            Shortcut for table view
    --index, -i            Row index or feature value for specific sample selection
    --index-feature, -f    Feature column name to search for index value
    --exp, -e              Field expressions with slicing syntax (e.g., 'res[:100],query[2:5]')
    --columns, -c          Select specific columns to display (e.g., 'query,sample')

Author: Generated with Claude Code
"""

import argparse
import sys
import re
from pathlib import Path
from datasets import load_dataset, load_from_disk
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from rich.panel import Panel

def parse_field_expressions(exp_str):
    """Parse field expressions like 'res[:100],query[2:5]' into a dict of field -> slice info"""
    if not exp_str:
        return {}
    
    field_slices = {}
    expressions = exp_str.split(',')
    
    for expr in expressions:
        expr = expr.strip()
        # Match pattern: field[start:end] or field[:end] or field[start:]
        match = re.match(r'^([^[]+)\[([^:]*):?([^]]*)\]$', expr)
        if match:
            field_name = match.group(1).strip()
            start_str = match.group(2).strip()
            end_str = match.group(3).strip()
            
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            
            field_slices[field_name] = (start, end)
    
    return field_slices

def truncate_field_value(value, slice_info):
    """Truncate field value based on slice info (start, end)"""
    if slice_info is None:
        return value
    
    start, end = slice_info
    value_str = str(value)
    
    if start is None and end is None:
        return value_str
    elif start is None:
        return value_str[:end]
    elif end is None:
        return value_str[start:]
    else:
        return value_str[start:end]

def main():
    parser = argparse.ArgumentParser(description="View HuggingFace dataset files")
    parser.add_argument("name", nargs="?", help="Dataset path/name")
    parser.add_argument("--samples", "-s", type=int, default=1, help="Number of samples to show (default: 1, -1 for full dataset)")
    parser.add_argument("--view", "-v", choices=["normal", "table"], default="normal", help="Display format: normal (default) or table")
    parser.add_argument("--table", "-t", action="store_const", const="table", dest="view", help="Display in table format (same as --view table)")
    parser.add_argument("--index", "-i", help="Index (row number) or feature value to show specific sample")
    parser.add_argument("--index-feature", "-f", help="Feature column name to search for the index value")
    parser.add_argument("--exp", "-e", help="Field expressions with slicing syntax (e.g., 'res[:100],query[2:5]')")
    parser.add_argument("--columns", "-c", help="Select specific columns to display (e.g., 'query,sample')")
    
    args = parser.parse_args()
    
    # Parse field expressions for truncation
    field_slices = parse_field_expressions(args.exp)
    
    # Parse selected columns
    selected_columns = None
    if args.columns:
        selected_columns = [col.strip() for col in args.columns.split(',')]
    
    # Use first positional argument as dataset name/path
    dataset_path = args.name
    if not dataset_path:
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            print("Error: Please provide a dataset path")
            sys.exit(1)
    
    console = Console()
    
    try:
        # Load the dataset
        console.print(f"[blue]Loading dataset: {dataset_path}[/blue]")
        
        # Check if it's a local path or HF dataset
        if Path(dataset_path).exists():
            path = Path(dataset_path)
            
            # If it's a directory, try load_from_disk first, then load_dataset
            if path.is_dir():
                console.print(f"[yellow]Loading from directory: {dataset_path}[/yellow]")
                try:
                    dataset = load_from_disk(dataset_path)
                except Exception:
                    console.print(f"[yellow]Trying load_dataset instead...[/yellow]")
                    dataset = load_dataset(dataset_path)
            else:
                # Determine file format based on extension
                extension = path.suffix.lower()
                
                if extension == '.json':
                    dataset = load_dataset("json", data_files=dataset_path)
                elif extension == '.jsonl':
                    dataset = load_dataset("json", data_files=dataset_path)
                elif extension == '.csv':
                    dataset = load_dataset("csv", data_files=dataset_path)
                elif extension == '.parquet':
                    dataset = load_dataset("parquet", data_files=dataset_path)
                elif extension == '.txt':
                    dataset = load_dataset("text", data_files=dataset_path)
                elif extension == '.tsv':
                    dataset = load_dataset("csv", data_files=dataset_path, delimiter='\t')
                else:
                    # Try to auto-detect format
                    console.print(f"[yellow]Unknown extension {extension}, attempting auto-detection...[/yellow]")
                    dataset = load_dataset("json", data_files=dataset_path)
        else:
            dataset = load_dataset(dataset_path)
        
        # Handle both Dataset and DatasetDict objects
        if hasattr(dataset, 'keys'):
            # It's a DatasetDict
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        else:
            # It's a Dataset object directly
            split_name = 'dataset'
            data = dataset
        
        console.print(f"[green]Dataset loaded successfully![/green]")
        console.print(f"[yellow]Split: {split_name}, Total samples: {len(data)}[/yellow]")
        
        # Handle index-based selection
        selected_samples = []
        if args.index is not None:
            if args.index_feature:
                # Search by feature value
                for i, sample in enumerate(data):
                    if args.index_feature in sample and str(sample[args.index_feature]) == str(args.index):
                        selected_samples.append((i, sample))
                
                if not selected_samples:
                    console.print(f"[red]Error: No samples found with {args.index_feature}={args.index}[/red]")
                    sys.exit(1)
            else:
                # Use as row index
                try:
                    index = int(args.index)
                    if index < 0:
                        index = len(data) + index  # Support negative indexing
                    if 0 <= index < len(data):
                        selected_samples.append((index, data[index]))
                    else:
                        console.print(f"[red]Error: Index {args.index} out of range (0-{len(data)-1})[/red]")
                        sys.exit(1)
                except ValueError:
                    console.print(f"[red]Error: Invalid index '{args.index}'. Must be a number when --index-feature is not specified.[/red]")
                    sys.exit(1)
        else:
            # Show first N samples (original behavior)
            if args.samples == -1:
                num_samples = len(data)
            else:
                num_samples = min(args.samples, len(data))
            
            for i in range(num_samples):
                selected_samples.append((i, data[i]))
        
        if args.view == "table":
            # Table format: all samples in one table
            table = Table(show_header=True, header_style="bold magenta")
            
            # Add sample index column
            table.add_column("Sample", style="yellow", justify="center")
            
            # Get all unique field names from the selected samples
            all_fields = set()
            for idx, sample in selected_samples:
                all_fields.update(sample.keys())
            
            # Filter columns if specified
            if selected_columns:
                all_fields = [col for col in selected_columns if col in all_fields]
            else:
                all_fields = sorted(all_fields)
            
            # Add columns for each field
            for field in all_fields:
                table.add_column(field, style="white")
            
            # Add rows for each selected sample
            for idx, sample in selected_samples:
                row_data = [str(idx + 1)]  # Use original dataset index + 1
                for field in all_fields:
                    value = sample.get(field, "")
                    # Apply truncation if specified
                    if field in field_slices:
                        value_str = truncate_field_value(value, field_slices[field])
                    else:
                        value_str = str(value)
                    row_data.append(value_str)
                table.add_row(*row_data)
            
            title = f"Dataset Samples ({len(selected_samples)} selected)"
            console.print(Panel(table, title=title, border_style="green"))
        else:
            # Normal format: each sample in its own panel
            for idx, sample in selected_samples:
                # Create a panel for each sample
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                
                # Filter fields if columns are specified
                fields_to_show = sample.items()
                if selected_columns:
                    fields_to_show = [(k, v) for k, v in sample.items() if k in selected_columns]
                
                for key, value in fields_to_show:
                    # Apply truncation if specified
                    if key in field_slices:
                        value_str = truncate_field_value(value, field_slices[key])
                    else:
                        # Convert value to string, handling various data types
                        if isinstance(value, (dict, list)):
                            value_str = JSON.from_data(value)
                        else:
                            value_str = str(value)
                    
                    table.add_row(key, value_str)
                
                console.print(Panel(table, title=f"Sample {idx + 1}", border_style="green"))
                console.print()
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

