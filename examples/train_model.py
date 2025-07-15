#!/usr/bin/env python3
"""
Script for training the semantic search model with custom training data.

Usage:
    python train_model.py [--data_path PATH] [--output_dir DIR] [--epochs N] [--batch_size N]
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import track

# Add parent directory to path to allow importing from scraper_llm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scraper_llm.search_assistant import SearchAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

def load_training_data(file_path: str) -> list:
    """Load training data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the semantic search model')
    parser.add_argument('--data_path', type=str, default='examples/training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_models',
                       help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    args = parser.parse_args()
    
    # Initialize assistant
    console.print("\n[bold blue]Initializing Search Assistant...[/]")
    assistant = SearchAssistant()
    
    # Load training data
    try:
        console.print(f"\n[bold]Loading training data from {args.data_path}...[/]")
        training_data = load_training_data(args.data_path)
        console.print(f"Loaded {len(training_data)} training examples")
    except Exception as e:
        console.print(f"[red]Error loading training data: {e}[/]")
        return
    
    # Add training examples
    console.print("\n[bold]Adding training examples...[/]")
    for example in track(training_data, description="Processing examples"):
        try:
            assistant.add_training_example(
                query=example['query'],
                preferred_results=example['preferred'],
                negative_results=example.get('negative', [])
            )
        except Exception as e:
            logger.warning(f"Error adding example for query '{example.get('query')}': {e}")
    
    # Train the model
    console.print("\n[bold green]Starting model training...[/]")
    try:
        model_path = assistant.fine_tune_model(
            output_path=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_steps=min(100, len(training_data) // 2),
            show_progress_bar=True
        )
        console.print(f"\n[bold green]✅ Model trained and saved to: {model_path}[/]")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'num_examples': len(training_data),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'model_path': str(model_path)
        }
        
        metadata_path = Path(args.output_dir) / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        console.print(f"Training metadata saved to: {metadata_path}")
        
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/]")
        raise

if __name__ == "__main__":
    main()
