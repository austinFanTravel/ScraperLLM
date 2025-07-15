"""Command-line interface for ScraperLLM."""
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import typer
from loguru import logger
from pydantic import HttpUrl
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .search import SerpAPISearcher, WebSearcher, SearchEngine, SearchResult, SearchIntent, EntityType
from .core.config import settings
from .core.logging import setup_logging
from .utils.content_processor import ContentProcessor, WebContent
from .utils.results import format_results

# Initialize the CLI app
app = typer.Typer(
    name="scraperllm",
    help="ScraperLLM CLI for web search and content extraction.",
    add_completion=False,
    no_args_is_help=True,
)

# Initialize console for rich output
console = Console()

# Output formats
class OutputFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    CONSOLE = "console"

# Search intents mapping
SEARCH_INTENT_MAP = {
    "person": SearchIntent.PERSON,
    "location": SearchIntent.LOCATION,
    "thing": SearchIntent.THING,
}

# Settings are already imported from core.config
console = Console()


def print_version(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[bold]ScraperLLM[/bold] v{__version__}")
        raise typer.Exit()


def format_entities(entities: List[Dict[str, Any]]) -> str:
    """Format entities for display."""
    if not entities:
        return ""
    return ", ".join(f"{e['text']} ({e['type'].lower()})" for e in entities)


def print_help():
    """Print help message with examples."""
    help_text = """\
[bold]ScraperLLM - Advanced Web Search and Content Extraction[/bold]

[bold]Examples:[/bold]
  # Basic search
  scraperllm search "Python programming"

  # Search for a person
  scraperllm search --person "Elon Musk"

  # Search for a location
  scraperllm search --location "Eiffel Tower"

  # Search with output to JSON file
  scraperllm search "Python" --output results.json

  # Search with custom max results
  scraperllm search "AI" --max-results 10

  # Search with a timeout
  scraperllm search "machine learning" --timeout 10

  # Search with summarization
  scraperllm search "Python" --summarize

[bold]Output Formats:[/bold]
  - console: Pretty-printed table (default)
  - json: JSON format
  - csv: CSV format

Use --help for more options and information.
"""
    console.print(help_text)
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=lambda: print_version(True),
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging.",
    ),
    help: bool = typer.Option(
        None,
        "--help",
        "-h",
        help="Show help message and exit.",
        callback=lambda: print_help(),
        is_eager=True,
    ),
):
    """ScraperLLM CLI entry point."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=log_level)


@app.command()
async def search(
    query: str = typer.Argument(
        None,
        help="Search query. If not provided, at least one of --person, --location, or --thing must be specified.",
    ),
    # Search intents
    person: Optional[str] = typer.Option(
        None,
        "--person",
        "-p",
        help="Search for information about a specific person.",
    ),
    location: Optional[str] = typer.Option(
        None,
        "--location",
        "-l",
        help="Search for a specific location or place.",
    ),
    thing: Optional[str] = typer.Option(
        None,
        "--thing",
        "-t",
        help="Search for a specific thing, object, or concept.",
    ),
    # Search parameters
    max_results: int = typer.Option(
        5,
        "--max-results",
        "-n",
        min=1,
        max=100,
        help="Maximum number of results to return (1-100).",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        "-t",
        min=1,
        help="Timeout in seconds for the search request.",
    ),
    # Output options
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Format is determined by file extension (.json, .csv).",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.CONSOLE,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (console, json, csv).",
    ),
    no_entities: bool = typer.Option(
        False,
        "--no-entities",
        help="Disable entity extraction from search results.",
    ),
    color: bool = typer.Option(
        sys.stdout.isatty(),
        "--color/--no-color",
        help="Enable/disable colored output. Defaults to auto-detection.",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        "-s",
        help="Generate a coherent summary from the top search results.",
    ),
    summarize_top: int = typer.Option(
        3,
        "--summarize-top",
        min=1,
        max=5,
        help="Number of top results to include in the summary (1-5).",
    ),
):
    """
    Search the web and return results with optional entity extraction and summarization.
    
    You can search using a general query or specify a specific intent (person, location, thing).
    """
    # Validate search parameters
    if not query and not any([person, location, thing]):
        console.print("[red]Error:[/] Either provide a query or specify a search intent (--person, --location, --thing)")
        raise typer.Exit(1)
    
    # Determine search intent and construct query
    intent = None
    if person:
        intent = SearchIntent.PERSON
        query = person
    elif location:
        intent = SearchIntent.LOCATION
        query = location
    elif thing:
        intent = SearchIntent.THING
        query = thing
    
    # Determine output format based on file extension if output is specified
    output_format = format
    if output:
        ext = output.suffix.lower()
        if ext == ".json":
            output_format = OutputFormat.JSON
        elif ext == ".csv":
            output_format = OutputFormat.CSV
    
    try:
        # Show search parameters
        with console.status("[bold green]Searching...") as status:
            try:
                # Initialize the search engine with SerpAPI
                searcher = SerpAPISearcher()
            except Exception as e:
                console.print(f"[red]Error initializing SerpAPI searcher: {e}[/red]")
                console.print("[yellow]Falling back to WebSearcher...[/yellow]")
                searcher = WebSearcher(extract_entities=not no_entities)
            
            # Execute the search
            status.update("[bold green]Executing search...")
            
            # Use async search if summarization is enabled, otherwise use sync search
            if summarize and hasattr(searcher, 'search_async'):
                results = await searcher.search_async(
                    query=query,
                    max_results=max_results,
                    timeout=timeout,
                    intent=intent,
                )
            else:
                results = searcher.search(
                    query=query,
                    max_results=max_results,
                    timeout=timeout,
                    intent=intent,
                )
            
            if not results:
                console.print("[yellow]No results found.[/]")
                return
            
            # Format and output results
            status.update("[bold green]Formatting results...")
            
            # Format results for display
            output_str = format_results(
                results=results,
                format_type=output_format,
                output_file=output,
                include_entities=not no_entities,
                color=color,
            )
            
            # Display search results in console
            if format == OutputFormat.CONSOLE:
                console.print(f"\n[bold]Search Results for: '{query}'[/]")
                console.print(output_str)
            
            # Add summarization if enabled
            if summarize and results:
                console.print("\n[bold blue]Analyzing search results to generate a comprehensive answer...[/]")
                
                # Limit to top N results for summarization
                top_results = results[:min(summarize_top, len(results))]
                
                async with ContentProcessor() as processor:
                    # Extract content from top results
                    contents: List[WebContent] = []
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Processing content...", total=len(top_results))
                        for result in top_results:
                            url = result.get('url') or result.get('link')
                            if url:
                                content = await processor.extract_content(url)
                                if content:
                                    # Add relevance score if available
                                    content.relevance = float(result.get('relevance', 0.0))
                                    contents.append(content)
                            progress.update(task, advance=1)
                    
                    # Generate and display summary after search results
                    if contents:
                        summary = await processor.generate_summary(query, contents)
                        
                        # Split summary into main content and sources
                        summary_parts = summary.split('\n\nSources:')
                        summary_text = summary_parts[0].strip()
                        
                        # Display summary in a clean panel
                        console.print("\n[bold green]Answer:[/]")
                        console.print(Panel(
                            Markdown(summary_text),
                            border_style="green",
                            expand=False
                        ))
                        
                        # Display sources separately if available
                        if len(summary_parts) > 1:
                            sources = summary_parts[1].strip()
                            if sources:
                                console.print("\n[bold]Sources:[/]")
                                for source in sources.split(', '):
                                    if source.strip():
                                        console.print(f"  • {source.strip()}")
                        
                        console.print("\n[dim]Note: This is an AI-generated summary based on the search results.[/]")
                    else:
                        console.print("[yellow]Warning: Could not extract content for summarization.[/]")
            
            # If not summarizing, just show the regular output
            elif not output and output_str:
                console.print(output_str)
    
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")
        if settings.DEBUG:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1) from e


# Add a default command that shows help
@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        print_help()


if __name__ == "__main__":
    app()
