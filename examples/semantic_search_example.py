"""
Example usage of the SemanticSearcher for semantic search.

This script demonstrates how to:
1. Create a semantic search index
2. Add documents to the index
3. Search for similar documents
4. Save and load the index
"""
import os
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add parent directory to path so we can import scraper_llm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scraper_llm.search import SemanticSearcher

# Set up console for pretty printing
console = Console()

def print_results(query: str, results: list):
    """Print search results in a nice table."""
    table = Table(title=f"Semantic Search Results for: '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Text", style="magenta")
    
    for result in results:
        # Truncate text for display
        text = result['text']
        if len(text) > 100:
            text = text[:97] + '...'
        table.add_row(
            f"{result['score']:.3f}",
            text
        )
    
    console.print(table)

def main():
    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast fox leaps above a sleeping hound.",
        "Apples are a popular fruit enjoyed around the world.",
        "Oranges are a citrus fruit rich in vitamin C.",
        "Dogs are known as man's best friend.",
        "Cats are independent animals that make great pets.",
        "The capital of France is Paris.",
        "Paris is known as the City of Light.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning is a type of machine learning that uses neural networks."
    ]
    
    # Initialize the semantic searcher with a smaller model
    console.print("[bold blue]Initializing SemanticSearcher...[/]")
    searcher = SemanticSearcher(model_name='all-MiniLM-L6-v2')
    
    # Add documents to the index
    console.print(f"[blue]Adding {len(documents)} documents to the index...[/]")
    searcher.add_documents(documents)
    
    # Build the index
    console.print("[blue]Building the search index...[/]")
    searcher.build_index(n_trees=10)
    
    # Perform some searches
    test_queries = [
        "fast animal",
        "fruits",
        "pets",
        "France",
        "machine learning"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold green]Searching for: '{query}'[/]")
        results = searcher.search(query, k=3)
        print_results(query, results)
    
    # Save the index
    save_dir = "./data/indices"
    console.print(f"\n[blue]Saving index to {save_dir}...[/]")
    os.makedirs(save_dir, exist_ok=True)
    searcher.save_index(save_dir)
    
    # Demonstrate loading the saved index
    console.print("\n[bold blue]Loading saved index...[/]")
    loaded_searcher = SemanticSearcher.load_index(save_dir, model_name='all-MiniLM-L6-v2')
    
    # Test the loaded index
    query = "fuzzy animal"
    console.print(f"\n[bold green]Searching loaded index for: '{query}'[/]")
    results = loaded_searcher.search(query, k=3)
    print_results(query, results)

if __name__ == "__main__":
    main()
