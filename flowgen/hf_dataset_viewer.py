#!/usr/bin/env python3
"""
HuggingFace Dataset Viewer

A command-line tool for viewing and inspecting HuggingFace datasets with flexible
display options and sample selection capabilities.

Features:
- Load datasets from local files or HuggingFace Hub
- Support for multiple file formats (JSON, JSONL, CSV, Parquet, TSV, TXT)
- Two display modes: normal (detailed panels) and table (compact grid)
- Flexible sample selection by index or feature value matching
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

Arguments:
    name                    Dataset path/name (local file/directory or HuggingFace dataset)
    --samples, -s          Number of samples to show (default: 1, -1 for all)
    --view, -v             Display format: 'normal' or 'table' (default: normal)
    --table, -t            Shortcut for table view
    --index, -i            Row index or feature value for specific sample selection
    --index-feature, -f    Feature column name to search for index value

Author: Generated with Claude Code
"""

import argparse
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from rich.panel import Panel

def main():
    parser = argparse.ArgumentParser(description="View HuggingFace dataset files")
    parser.add_argument("name", nargs="?", help="Dataset path/name")
    parser.add_argument("--samples", "-s", type=int, default=1, help="Number of samples to show (default: 1, -1 for full dataset)")
    parser.add_argument("--view", "-v", choices=["normal", "table"], default="normal", help="Display format: normal (default) or table")
    parser.add_argument("--table", "-t", action="store_const", const="table", dest="view", help="Display in table format (same as --view table)")
    parser.add_argument("--index", "-i", help="Index (row number) or feature value to show specific sample")
    parser.add_argument("--index-feature", "-f", help="Feature column name to search for the index value")
    
    args = parser.parse_args()
    
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
            
            # Add columns for each field
            for field in sorted(all_fields):
                table.add_column(field, style="white")
            
            # Add rows for each selected sample
            for idx, sample in selected_samples:
                row_data = [str(idx + 1)]  # Use original dataset index + 1
                for field in sorted(all_fields):
                    value = sample.get(field, "")
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
                
                for key, value in sample.items():
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

