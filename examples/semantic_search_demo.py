"""
Semantic Search Demo for ScraperLLM

This script demonstrates how to use the SemanticSearchTool to:
1. Create and manage a semantic search index
2. Perform semantic searches
3. Fine-tune the model on custom data
4. Evaluate search quality
5. Combine semantic and keyword search

Prerequisites:
- Install required packages: pip install -r requirements.txt
- Set up your .env file with necessary API keys
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Add parent directory to path to import scraper_llm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scraper_llm.utils.semantic_search_utils import SemanticSearchTool, evaluate_search_quality, hybrid_search
from scraper_llm.search import SerpAPISearcher
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up console for pretty printing
console = Console()

def print_results(query: str, results: List[Dict[str, Any]], show_metadata: bool = False) -> None:
    """Print search results in a nice table."""
    if not results:
        console.print("[yellow]No results found.[/]")
        return
    
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Text", style="magenta")
    
    if show_metadata:
        table.add_column("Metadata", style="blue")
    
    for result in results:
        # Truncate text for display
        text = result['text']
        if len(text) > 100:
            text = text[:97] + '...'
        
        # Format score
        score = f"{result['score']:.3f}"
        
        if show_metadata and 'metadata' in result and result['metadata']:
            metadata = ", ".join(f"{k}:{v}" for k, v in result['metadata'].items())
            if len(metadata) > 50:
                metadata = metadata[:47] + '...'
            table.add_row(score, text, metadata)
        else:
            table.add_row(score, text)
    
    console.print(table)

def demo_basic_search():
    """Demonstrate basic semantic search functionality."""
    console.print("[bold blue]=== Basic Semantic Search Demo ===[/]")
    
    # Sample documents
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning is transforming industries worldwide.",
        "Python is a popular programming language for data science.",
        "The Great Wall of China is one of the wonders of the world.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Paris is known as the City of Light and is famous for its art and culture.",
        "Natural language processing enables computers to understand human language.",
        "The Louvre Museum in Paris houses the Mona Lisa painting.",
        "Artificial intelligence is revolutionizing healthcare and other sectors.",
        "The Eiffel Tower was completed in 1889 for the World's Fair."
    ]
    
    # Initialize the semantic searcher
    searcher = SemanticSearchTool(model_name="all-MiniLM-L6-v2")
    
    # Add documents to the index
    console.print("[blue]Adding documents to the index...[/]")
    searcher.add_documents(documents)
    
    # Build the index
    console.print("[blue]Building the search index...[/]")
    searcher.build_index(n_trees=10)
    
    # Perform some searches
    test_queries = [
        "famous landmarks in Paris",
        "machine learning technologies",
        "programming languages"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold green]Searching for: '{query}'[/]")
        results = searcher.search(query, k=3)
        print_results(query, results)
    
    return searcher

def demo_fine_tuning(searcher: SemanticSearchTool):
    """Demonstrate fine-tuning the semantic search model."""
    console.print("\n[bold blue]=== Model Fine-Tuning Demo ===[/]")
    
    # Sample training data (in a real scenario, you'd have more and better examples)
    train_examples = [
        {
            'query': 'AI in healthcare',
            'pos': 'Artificial intelligence is being used to improve healthcare outcomes.',
            'neg': 'The Eiffel Tower is a famous landmark in Paris.'
        },
        {
            'query': 'programming languages',
            'pos': 'Python and JavaScript are popular programming languages.',
            'neg': 'The Great Wall of China is an ancient structure.'
        },
        {
            'query': 'Paris landmarks',
            'pos': 'The Eiffel Tower and the Louvre are famous landmarks in Paris.',
            'neg': 'Machine learning is a branch of artificial intelligence.'
        }
    ]
    
    # Fine-tune the model
    output_path = "./models/fine_tuned_model"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    console.print("[blue]Fine-tuning the model (this may take a few minutes)...[/]")
    searcher.fine_tune(
        train_examples=train_examples,
        output_path=output_path,
        epochs=2,  # Use more epochs in practice
        batch_size=2
    )
    
    console.print(f"[green]Model fine-tuned and saved to {output_path}[/]")
    
    # Reload the fine-tuned model
    searcher = SemanticSearchTool(model_name=output_path)
    
    # Test with a query that should benefit from fine-tuning
    query = "AI applications"
    console.print(f"\n[bold green]Searching after fine-tuning: '{query}'[/]")
    results = searcher.search(query, k=3)
    print_results(query, results)
    
    return searcher

async def demo_hybrid_search():
    """Demonstrate hybrid search combining semantic and keyword search."""
    console = Console()
    console.print("\n[bold blue]=== Hybrid Search Demo ===[/]")
    
    # Initialize searchers
    semantic_searcher = SemanticSearchTool(model_name="all-MiniLM-L6-v2")
    
    # Initialize SerpAPI searcher
    try:
        keyword_searcher = SerpAPISearcher()
    except ValueError as e:
        console.print(f"[yellow]Warning: {e}[/]")
        console.print("Skipping hybrid search demo. Make sure to set SERPAPI_KEY in your .env file.")
        return
    
    # Add some sample documents
    documents = [
        {
            'text': 'The Eiffel Tower is located in Paris, France.',
            'source': 'wikipedia',
            'category': 'landmarks'
        },
        {
            'text': 'Machine learning is transforming industries worldwide.',
            'source': 'tech_blog',
            'category': 'technology'
        },
        {
            'text': 'Python is a popular programming language for data science.',
            'source': 'programming_guide',
            'category': 'programming'
        },
        {
            'text': 'The Louvre Museum in Paris houses the Mona Lisa painting.',
            'source': 'art_history',
            'category': 'art'
        }
    ]
    
    # Add documents to the semantic index
    semantic_searcher.add_documents(documents)
    semantic_searcher.build_index()
    
    # Perform a hybrid search
    query = "famous places in Paris"
    console.print(f"[bold green]Performing hybrid search for: '{query}'[/]")
    
    # Get semantic results
    semantic_results = semantic_searcher.search(query, k=3)
    console.print("\n[underline]Semantic Search Results:[/]")
    print_results(query, semantic_results, show_metadata=True)
    
    # Get keyword results (using SerpAPI)
    try:
        console.print("\n[underline]Keyword Search Results (from SerpAPI):[/]")
        
        # Execute the async search
        if hasattr(keyword_searcher, 'search_async'):
            keyword_results = await keyword_searcher.search_async(query, max_results=3)
            
            # Convert to common format for display
            formatted_keyword_results = [
                {
                    'text': f"{getattr(res, 'title', '')}: {getattr(res, 'snippet', '')}",
                    'score': 1.0 - (i / len(keyword_results)) if keyword_results else 0,
                    'metadata': {
                        'source': 'serpapi',
                        'url': getattr(res, 'url', '')
                    }
                }
                for i, res in enumerate(keyword_results or [])
            ]
            print_results(query, formatted_keyword_results, show_metadata=True)
            
            # Perform hybrid search
            console.print("\n[underline]Hybrid Search Results:[/]")
            hybrid_results = await hybrid_search(
                semantic_searcher=semantic_searcher,
                keyword_searcher=keyword_searcher,
                query=query,
                k=3,
                alpha=0.7  # Weight for semantic search
            )
            print_results(query, hybrid_results, show_metadata=True)
        else:
            console.print("[yellow]Warning: Keyword searcher does not support async search[/]")
            
    except Exception as e:
        console.print(f"[yellow]Warning: Could not perform keyword search: {e}[/]")
        console.print("Make sure you have set up SerpAPI and have a valid API key in your .env file.")
        import traceback
        traceback.print_exc()

async def main():
    """Run all demos."""
    console = Console()
    
    try:
        # Run basic search demo
        demo_basic_search()
        
        # Run hybrid search demo
        await demo_hybrid_search()
        
        console.print("\n[bold green]Demo completed successfully![/]")
    except Exception as e:
        console.print(f"[red]Error during demo: {e}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
