#!/usr/bin/env python3
"""
Interactive CLI for the Semantic Search Assistant with Summarization
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.markdown import Markdown

import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper_llm.search_assistant import SearchAssistant
from scraper_llm.utils.semantic_search_utils import SemanticSearchTool
from scraper_llm.utils.content_processor import ContentProcessor, WebContent

# Initialize console for rich output
console = Console()

class SearchCLI:
    def __init__(self, model_path=None):
        """Initialize the search CLI
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.console = Console()
        self.current_query = ""
        self.search_history = []
        self.model_path = model_path or "./fine_tuned_models"
        
        # Initialize search assistant
        self._init_assistant()
    
    def _init_assistant(self):
        """Initialize the search assistant with the fine-tuned model"""
        try:
            self.console.print("[yellow]Loading search assistant...[/yellow]")
            
            # Check if model path exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.console.print(f"[yellow]Model not found at {model_path}, using default model[/yellow]")
                self.assistant = SearchAssistant()
                return
                
            # Initialize SearchAssistant with the custom model path
            self.assistant = SearchAssistant(
                model_name=str(model_path.absolute())
            )
            self.console.print("[green]✓ Loaded fine-tuned model![/green]\n")
                
        except Exception as e:
            self.console.print(f"[red]Error initializing search assistant: {e}[/red]")
            self.console.print("[yellow]Falling back to default model...[/yellow]")
            self.assistant = SearchAssistant()
    
    async def search_loop(self):
        """Main search loop"""
        while True:
            self.console.print("\n" + "="*50)
            self.console.print("[bold blue]Semantic Search Assistant[/bold blue]")
            self.console.print("1. New search")
            self.console.print("2. View search history")
            self.console.print("3. Exit")
            
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3"])
            
            if choice == "1":
                await self._new_search()
            elif choice == "2":
                self._view_history()
            else:
                self.console.print("[yellow]Goodbye![/yellow]")
                break
    
    async def _new_search(self):
        """Handle a new search"""
        self.console.print("\n[bold]New Search[/bold]")
        query = Prompt.ask("Enter your search query")
        
        # Get search parameters
        num_results = IntPrompt.ask(
            "Number of results (default: 5)", 
            default=5,
            show_default=True
        )
        
        min_relevance = FloatPrompt.ask(
            "Minimum relevance (0.0 to 1.0, default: 0.3)",
            default=0.3,
            show_default=True
        )
        
        # Ask if user wants summarization
        summarize = Confirm.ask(
            "\n[bold]Would you like to generate a summary from the top results?[/bold]",
            default=False
        )
        
        summarize_top = 3
        if summarize:
            while True:
                try:
                    value = IntPrompt.ask(
                        "Number of top results to summarize (1-5, default: 3)",
                        default=3
                    )
                    if 1 <= value <= 5:
                        summarize_top = value
                        break
                    self.console.print("[red]Please enter a number between 1 and 5[/red]")
                except ValueError:
                    self.console.print("[red]Please enter a valid number[/red]")
        
        self.current_query = query
        
        # Perform search
        self.console.print(f"\n[bold]Searching for:[/bold] {query}")
        self.console.print("[yellow]This may take a moment...[/yellow]")
        
        try:
            # Await the async search call
            results = await self.assistant.search(
                query=query,
                num_results=num_results,
                min_relevance=min_relevance
            )
            
            # Save to history
            self.search_history.append({
                "query": query,
                "results": results,
                "num_results": num_results,
                "min_relevance": min_relevance
            })
            
            # Display results
            self._display_results(results)
            
            # Generate and display summary if requested
            if summarize and results:
                await self._generate_summary(query, results[:summarize_top])
            
            # Ask for feedback
            self._get_feedback(results)
            
        except Exception as e:
            self.console.print(f"[red]Error performing search: {e}[/red]")
    
    def _display_results(self, results):
        """Display search results in a formatted table"""
        if not results:
            self.console.print("[yellow]No results found.[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", width=3)
        table.add_column("Title", width=50)
        table.add_column("URL", width=50)
        table.add_column("Relevance", width=10)
        table.add_column("Domain", width=20)
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No snippet available')
            relevance = result.get('relevance_score', result.get('relevance', 0))
            domain = result.get('domain', self._extract_domain(url) if url != 'No URL' else 'N/A')
            
            # Truncate long titles and URLs for display
            display_title = (title[:47] + '...') if len(title) > 50 else title
            display_url = (url[:47] + '...') if len(url) > 50 else url
            
            table.add_row(
                str(i),
                display_title,
                display_url,
                f"{float(relevance):.2f}",
                domain
            )
        
        self.console.print(table)
        
        # Show details for a specific result
        while True:
            try:
                choice = Prompt.ask(
                    "\nEnter a result number to see details, or press Enter to continue",
                    default=""
                )
                
                if not choice:
                    break
                    
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    self._show_result_details(results[idx])
                else:
                    self.console.print("[red]Invalid result number[/red]")
                    
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")
    
    def _extract_domain(self, url):
        """Extract domain from URL"""
        if not url or url == 'No URL':
            return 'N/A'
        try:
            # Remove protocol and www. if present
            domain = url.split('//')[-1].split('/')[0]
            domain = domain.replace('www.', '')
            # Take only the main domain (e.g., 'example.com' from 'sub.example.com')
            return '.'.join(domain.split('.')[-2:])
        except Exception:
            return 'N/A'
    
    def _show_result_details(self, result):
        """Show detailed information about a search result"""
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        snippet = result.get('snippet', 'No snippet available')
        relevance = result.get('relevance_score', result.get('relevance', 0))
        
        panel = Panel(
            f"[bold]URL:[/bold] {url}\n\n"
            f"[bold]Relevance:[/bold] {relevance:.2f}\n\n"
            f"[bold]Snippet:[/bold] {snippet}",
            title=title,
            title_align="left",
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def _get_feedback(self, results):
        """Get feedback on search results"""
        if not results:
            return
            
        self.console.print("\n[bold]Search Feedback[/bold]")
        feedback = Prompt.ask(
            "Were these results helpful?",
            choices=["y", "n", "s"],
            show_choices=True
        )
        
        if feedback == "y":
            self.console.print("[green]Great! We'll use this to improve future searches.[/green]")
        elif feedback == "n":
            self.console.print("[yellow]We're sorry to hear that. What could be better?[/yellow]")
            improvement = Prompt.ask("Your feedback")
            # Here you could log this feedback for model improvement
        else:
            self.console.print("[yellow]Thanks for your feedback![/yellow]")
    
    async def _generate_summary(self, query: str, results: List[dict]):
        """Generate and display a summary of the top search results"""
        self.console.print("\n[bold blue]Analyzing search results to generate a comprehensive answer...[/]")
        
        async with ContentProcessor() as processor:
            # Extract content from top results
            contents: List[WebContent] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing content...", total=len(results))
                for result in results:
                    url = result.get('url') or result.get('link')
                    if url:
                        try:
                            content = await processor.extract_content(url, verify_ssl=False)
                            if content:
                                content.relevance = float(result.get('relevance', 0.0))
                                contents.append(content)
                                self.console.print(f"[green]✓ Processed: {url}[/]")
                            else:
                                self.console.print(f"[yellow]⚠️  No content extracted from: {url}[/]")
                        except Exception as e:
                            self.console.print(f"[red]❌ Error processing {url}: {str(e)}[/]")
                    progress.update(task, advance=1)
            
            # Generate and display summary if we have content
            if contents:
                try:
                    summary = await processor.generate_summary(query, contents)
                    
                    # Split summary into main content and sources
                    summary_parts = summary.split('\n\nSources:')
                    summary_text = summary_parts[0].strip()
                    
                    # Display the answer in a clean, formatted panel
                    self.console.print("\n[bold green]Answer:[/]")
                    
                    # Create a panel with clean, wrapped text
                    panel = Panel(
                        summary_text,
                        border_style="green",
                        expand=False,
                        width=min(80, self.console.width - 4),
                        padding=(1, 2)
                    )
                    self.console.print(panel)
                    
                    # Display sources if available
                    if len(summary_parts) > 1 and summary_parts[1].strip():
                        self.console.print("\n[bold]Sources:[/]")
                        for source in summary_parts[1].split(','):
                            if source.strip():
                                self.console.print(f"  • {source.strip()}")
                    
                    self.console.print("\n[dim]Note: This is an AI-generated summary based on the search results.[/]")
                except Exception as e:
                    self.console.print(f"[red]Error generating summary: {str(e)}[/]")
                    self.console.print("[yellow]Showing raw content instead:[/]")
                    for content in contents:
                        self.console.print(f"\n[bold]{content.title}[/] ({content.url})")
                        self.console.print(content.text[:500] + ("..." if len(content.text) > 500 else ""))
            else:
                self.console.print("[yellow]⚠️  Could not extract any content for summarization.[/]")
                self.console.print("This could be due to:")
                self.console.print("  • Websites blocking automated access")
                self.console.print("  • Network connectivity issues")
                self.console.print("  • Paywalls or login requirements")
                self.console.print("\nTry again with different search terms or check your internet connection.")
    
    def _view_history(self):
        """View search history"""
        if not self.search_history:
            self.console.print("[yellow]No search history.[/yellow]")
            return
            
        self.console.print("\n[bold]Search History[/bold]")
        for i, search in enumerate(self.search_history, 1):
            self.console.print(f"{i}. {search['query']} ({len(search['results'])} results)")
        
        while True:
            choice = Prompt.ask(
                "\nEnter a number to view details, or press Enter to go back",
                default=""
            )
            
            if not choice:
                break
                
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.search_history):
                    search = self.search_history[idx]
                    self.console.print(f"\n[bold]Query:[/bold] {search['query']}")
                    self.console.print(f"[bold]Date:[/bold] {search.get('timestamp', 'N/A')}")
                    self._display_results(search['results'])
                else:
                    self.console.print("[red]Invalid selection[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

def main():
    parser = argparse.ArgumentParser(description='Search CLI for ScraperLLM')
    parser.add_argument(
        '--model', 
        type=str, 
        default="./fine_tuned_models",
        help="Path to fine-tuned model (default: ./fine_tuned_models)"
    )
    args = parser.parse_args()
    
    # Create and run the CLI
    cli = SearchCLI(model_path=args.model if os.path.exists(args.model) else None)
    import asyncio
    asyncio.run(cli.search_loop())

if __name__ == "__main__":
    main()
