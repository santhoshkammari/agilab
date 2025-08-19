#!/usr/bin/env python3

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
    parser.add_argument("--samples", "-s", type=int, default=1, help="Number of samples to show (default: 2)")
    parser.add_argument("--view", "-v", choices=["normal", "table"], default="normal", help="Display format: normal (default) or table")
    
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
        
        # Show first N samples
        num_samples = min(args.samples, len(data))
        
        if args.view == "table":
            # Table format: all samples in one table
            table = Table(show_header=True, header_style="bold magenta")
            
            # Add sample index column
            table.add_column("Sample", style="yellow", justify="center")
            
            # Get all unique field names from the samples
            all_fields = set()
            samples_data = []
            for i in range(num_samples):
                sample = data[i]
                samples_data.append(sample)
                all_fields.update(sample.keys())
            
            # Add columns for each field
            for field in sorted(all_fields):
                table.add_column(field, style="white")
            
            # Add rows for each sample
            for i, sample in enumerate(samples_data):
                row_data = [str(i + 1)]
                for field in sorted(all_fields):
                    value = sample.get(field, "")
                    value_str = str(value)
                    row_data.append(value_str)
                table.add_row(*row_data)
            
            console.print(Panel(table, title=f"Dataset Samples (1-{num_samples})", border_style="green"))
        else:
            # Normal format: each sample in its own panel
            for i in range(num_samples):
                sample = data[i]
                
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
                
                console.print(Panel(table, title=f"Sample {i + 1}", border_style="green"))
                console.print()
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

