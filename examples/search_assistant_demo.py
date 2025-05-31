"""
Search Assistant Demo

This script demonstrates how to use the SearchAssistant class to perform
intelligent searches, filter results, and improve search quality over time.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add parent directory to path to import scraper_llm
sys.path.append(str(Path(__file__).parent.parent))

from scraper_llm.search_assistant import SearchAssistant

# Load environment variables
load_dotenv()

# Set up console for pretty printing
console = Console()

def print_search_results(query: str, results: List[Dict]):
    """Print search results in a nice table."""
    if not results:
        console.print("[yellow]No results found.[/]")
        return
    
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Relevance", style="green")
    table.add_column("Domain", style="blue")
    
    for result in results:
        # Truncate long titles
        title = result['title']
        if len(title) > 50:
            title = title[:47] + '...'
            
        table.add_row(
            str(result['rank']),
            title,
            f"{result['relevance_score']:.3f}",
            result.get('domain', 'N/A')
        )
    
    console.print(table)

def main():
    """Run the search assistant demo."""
    try:
        # Initialize the search assistant
        console.print("[bold blue]Initializing Search Assistant...[/]")
        assistant = SearchAssistant(
            model_name="all-mpnet-base-v2",
            data_dir="./data/search_assistant",
            use_gpu=False  # Set to True if you have a GPU
        )
        
        # Main interaction loop
        while True:
            console.print("\n[bold]Search Assistant Menu:[/]")
            console.print("1. Perform a search")
            console.print("2. View search history")
            console.print("3. Add training example")
            console.print("4. Fine-tune the model")
            console.print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                # Perform a search
                query = input("\nEnter your search query: ").strip()
                if not query:
                    console.print("[yellow]Query cannot be empty.[/]")
                    continue
                
                # Get domain filters
                domains = input("Filter by domains (comma-separated, leave empty for none): ").strip()
                domains = [d.strip() for d in domains.split(',')] if domains else None
                
                # Get minimum relevance
                min_relevance = 0.0
                while True:
                    rel_input = input("Minimum relevance score (0.0-1.0, default 0.0): ").strip()
                    if not rel_input:
                        break
                    try:
                        min_relevance = float(rel_input)
                        if 0 <= min_relevance <= 1:
                            break
                        console.print("[yellow]Please enter a value between 0.0 and 1.0[/]")
                    except ValueError:
                        console.print("[yellow]Please enter a valid number.[/]")
                
                # Perform the search
                with console.status("Searching...", spinner="dots"):
                    results = assistant.search(
                        query=query,
                        num_results=10,
                        min_relevance=min_relevance,
                        domains=domains,
                        export_excel=True
                    )
                
                # Display results
                print_search_results(query, results)
                
                # Ask for feedback
                if results:
                    feedback = input("\nWas this search helpful? (y/n): ").strip().lower()
                    if feedback == 'n':
                        # Get preferred results
                        console.print("\nPlease provide an example of a better result:")
                        better_result = input("Enter a relevant URL or text: ").strip()
                        if better_result:
                            assistant.add_training_example(
                                query=query,
                                preferred_results=[better_result],
                                negative_results=[r for r in results if r['relevance_score'] < 0.5]
                            )
                            console.print("[green]Thank you for your feedback![/]")
            
            elif choice == '2':
                # View search history
                history = assistant.get_search_history(limit=10)
                if not history:
                    console.print("[yellow]No search history found.[/]")
                    continue
                
                table = Table(title="Recent Search History")
                table.add_column("Query", style="cyan")
                table.add_column("Timestamp", style="green")
                table.add_column("Results", style="magenta")
                
                for entry in history:
                    table.add_row(
                        entry['query'][:50] + ('...' if len(entry['query']) > 50 else ''),
                        entry.get('timestamp', 'N/A'),
                        str(entry.get('num_results', 0))
                    )
                
                console.print(table)
            
            elif choice == '3':
                # Add training example
                console.print("\n[bold]Add Training Example[/]")
                query = input("Enter the search query: ").strip()
                
                preferred = []
                console.print("\nEnter preferred results (one per line, leave empty when done):")
                while True:
                    result = input(f"Preferred result {len(preferred) + 1}: ").strip()
                    if not result:
                        break
                    preferred.append(result)
                
                negative = []
                console.print("\nEnter negative results (one per line, leave empty if none):")
                while True:
                    result = input(f"Negative result {len(negative) + 1}: ").strip()
                    if not result:
                        break
                    negative.append(result)
                
                if query and (preferred or negative):
                    assistant.add_training_example(
                        query=query,
                        preferred_results=[{'text': p} for p in preferred],
                        negative_results=[{'text': n} for n in negative]
                    )
                    console.print("[green]Training example added successfully![/]")
                else:
                    console.print("[yellow]Query and at least one preferred or negative result are required.[/]")
            
            elif choice == '4':
                # Fine-tune the model
                if not assistant.training_data:
                    console.print("[yellow]No training data available. Add examples first.[/]")
                    continue
                
                console.print(f"\n[bold]Fine-tuning model with {len(assistant.training_data)} examples...[/]")
                try:
                    output_path = assistant.fine_tune_model(
                        epochs=3,
                        batch_size=8,
                        show_progress_bar=True
                    )
                    console.print(f"[green]Model fine-tuned and saved to: {output_path}[/]")
                except Exception as e:
                    console.print(f"[red]Error during fine-tuning: {e}[/]")
            
            elif choice == '5':
                # Exit
                console.print("[bold blue]Goodbye![/]")
                break
            
            else:
                console.print("[yellow]Invalid choice. Please enter a number between 1 and 5.[/]")
    
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user.[/]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/]")
        import traceback
        console.print(traceback.format_exc())
    finally:
        # Ensure data is saved before exiting
        if 'assistant' in locals():
            assistant._save_data()

if __name__ == "__main__":
    # Add the missing List and Dict imports at runtime
    from typing import List, Dict
    main()
