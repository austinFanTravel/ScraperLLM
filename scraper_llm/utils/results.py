"""Utilities for handling and exporting search results."""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from rich.console import Console
from rich.table import Table

from ..search.base import Entity, EntityType, SearchResult, SearchIntent


def format_results(
    results: List[SearchResult],
    format_type: str = "console",
    output_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional[str]:
    """Format search results in the specified format.
    
    Args:
        results: List of search results
        format_type: Output format (console, json, csv)
        output_file: Optional file path to save results
        **kwargs: Additional formatting options
            - include_entities: Whether to include entities in the output
            - color: Whether to use color in console output
            
    Returns:
        Formatted results as a string if not writing to file, else None
    """
    format_type = format_type.lower()
    include_entities = kwargs.get("include_entities", True)
    
    if format_type == "json":
        return _export_to_json(results, output_file, include_entities)
    elif format_type == "csv":
        return _export_to_csv(results, output_file, include_entities)
    else:  # console
        return _format_for_console(results, include_entities, **kwargs)


def _export_to_json(
    results: List[SearchResult],
    output_file: Optional[Union[str, Path]] = None,
    include_entities: bool = True,
) -> Optional[str]:
    """Export results to JSON format.
    
    Args:
        results: List of search results
        output_file: Optional file path to save results
        include_entities: Whether to include entities in the output
        
    Returns:
        JSON string if output_file is None, else None
    """
    result_data = [result.dict() for result in results]
    
    # Clean up the data
    for item in result_data:
        # Convert URL to string
        item["url"] = str(item["url"])
        
        # Convert datetime to ISO format
        if isinstance(item.get("timestamp"), datetime):
            item["timestamp"] = item["timestamp"].isoformat()
        
        # Optionally remove entities
        if not include_entities and "entities" in item:
            del item["entities"]
    
    json_str = json.dumps(result_data, indent=2, ensure_ascii=False)
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json_str, encoding="utf-8")
        logger.info(f"Results exported to {output_file}")
        return None
    
    return json_str


def _export_to_csv(
    results: List[SearchResult],
    output_file: Optional[Union[str, Path]] = None,
    include_entities: bool = True,
) -> Optional[str]:
    """Export results to CSV format.
    
    Args:
        results: List of search results
        output_file: Optional file path to save results
        include_entities: Whether to include entities in the output
        
    Returns:
        CSV string if output_file is None, else None
    """
    if not results:
        return ""
    
    # Prepare data for CSV
    rows = []
    for result in results:
        row = {
            "title": result.title,
            "url": str(result.url),
            "source": result.source,
            "timestamp": result.timestamp.isoformat() if result.timestamp else "",
            "snippet": result.snippet,
        }
        
        if include_entities and result.entities:
            row["entities"] = "; ".join(
                f"{e['text']} ({e['type']})" for e in result.entities
            )
        
        rows.append(row)
    
    # Get all unique fieldnames
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        logger.info(f"Results exported to {output_file}")
        return None
    else:
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
        return output.getvalue()


def _format_for_console(
    results: List[SearchResult],
    include_entities: bool = True,
    **kwargs,
) -> str:
    """Format results for console output.
    
    Args:
        results: List of search results
        include_entities: Whether to include entities in the output
        **kwargs: Additional options
            - color: Whether to use color in the output
            
    Returns:
        Formatted string for console output
    """
    use_color = kwargs.get("color", True)
    console = Console(color_system="auto" if use_color else None)
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return ""
    
    # Create a table for the results
    table = Table(
        title=f"Search Results ({len(results)})",
        show_header=True,
        header_style="bold magenta",
        box=None if use_color else None,
    )
    
    # Add columns
    table.add_column("#", width=4)
    table.add_column("Title", style="cyan", no_wrap=True)
    table.add_column("URL", style="blue", no_wrap=True, overflow="ellipsis")
    
    if include_entities:
        table.add_column("Entities", style="green", no_wrap=True, max_width=30)
    
    # Add rows
    for i, result in enumerate(results, 1):
        url = str(result.url)
        if len(url) > 40:
            url = url[:20] + "..." + url[-20:]
            
        row = [
            f"[yellow]{i}[/yellow]",
            result.title,
            url,
        ]
        
        if include_entities and result.entities:
            entity_str = ", ".join(
                f"[green]{e['text']}[/green]" for e in result.entities[:3]
            )
            if len(result.entities) > 3:
                entity_str += f" +{len(result.entities) - 3} more"
            row.append(entity_str)
        
        table.add_row(*row)
    
    # Print the table
    console.print(table)
    
    # Print entity legend if needed
    if include_entities and any(r.entities for r in results):
        console.print("\n[bold]Entity Types:[/bold]")
        console.print("  [green]PERSON[/green], [blue]LOCATION[/blue], [magenta]ORGANIZATION[/magenta], etc.")
    
    # Return empty string since we're printing directly to console
    return ""
