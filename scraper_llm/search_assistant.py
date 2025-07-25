"""
Search Assistant for ScraperLLM

A comprehensive search assistant that combines semantic and keyword search
with machine learning to provide relevant search results.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from loguru import logger

# Import required modules
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Some features will be disabled.")

# Import from local modules
from .search import SerpAPISearcher
from .utils.semantic_search_utils import SemanticSearchTool, hybrid_search


class SearchAssistant:
    """
    An intelligent search assistant that combines semantic and keyword search
    with the ability to learn from user feedback.
    """

    class SearchAssistant:
        """An intelligent search assistant that combines semantic and keyword search
        with the ability to learn from user feedback.
        """
        
        # Target social media sites for specialized searches
        SOCIAL_MEDIA_SITES = [
            "site:facebook.com",
            "site:instagram.com",
            "site:twitter.com",
            "site:tiktok.com"
        ]
        
        # Number of results to fetch per social media site
        RESULTS_PER_SITE = 3
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        data_dir: str = "./data/search_assistant",
        use_gpu: bool = False
    ):
        """
        Initialize the search assistant.
        
        Args:
            model_name: Name of the sentence transformer model to use
            data_dir: Directory to store search history and models
            use_gpu: Whether to use GPU if available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SearchAssistant. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.search_history = []
        self.training_data = []
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize searchers
        self.semantic_searcher = SemanticSearchTool(
            model_name=model_name,
            use_gpu=use_gpu
        )
        
        self.keyword_searcher = SerpAPISearcher()
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load search history and training data from disk."""
        # Load search history
        history_file = self.data_dir / "search_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.search_history = json.load(f)
                logger.info(f"Loaded {len(self.search_history)} search history entries")
            except Exception as e:
                logger.warning(f"Failed to load search history: {e}")
        
        # Load training data
        training_file = self.data_dir / "training_data.json"
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    self.training_data = json.load(f)
                logger.info(f"Loaded {len(self.training_data)} training examples")
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")
    
    def _save_data(self):
        """Save search history and training data to disk."""
        try:
            # Save search history
            with open(self.data_dir / "search_history.json", 'w') as f:
                json.dump(self.search_history, f, indent=2)
            
            # Save training data
            with open(self.data_dir / "training_data.json", 'w') as f:
                json.dump(self.training_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    async def _keyword_search_wrapper(self, query: str, k: int = 10, **kwargs) -> List[Dict]:
        """
        Wrapper method to make SerpAPISearcher compatible with hybrid_search.
        
        Args:
            query: The search query
            k: Maximum number of results to return (synonym for max_results)
            **kwargs: Additional keyword arguments (including max_results for backward compatibility)
            
        Returns:
            List of search results in the format expected by hybrid_search
        """
        # Handle both 'k' and 'max_results' parameters
        max_results = kwargs.get('max_results', k)
        
        try:
            # Call the actual keyword search method
            if hasattr(self.keyword_searcher, 'search_async'):
                results = await self.keyword_searcher.search_async(query, max_results=max_results)
            else:
                results = self.keyword_searcher.search(query, max_results=max_results)
            
            # Format results to match expected format
            formatted_results = []
            if results:
                for i, result in enumerate(results, 1):
                    # Handle different result formats
                    if isinstance(result, dict):
                        formatted_result = {
                            'text': result.get('snippet', result.get('text', '')),
                            'metadata': {
                                'title': result.get('title', ''),
                                'url': result.get('link', result.get('url', '')),
                                'source': result.get('source', 'keyword_search')
                            },
                            'score': 1.0 - (i * 0.05)  # Slight score decay
                        }
                    else:
                        # Handle case where result is an object with attributes
                        formatted_result = {
                            'text': getattr(result, 'snippet', getattr(result, 'text', '')),
                            'metadata': {
                                'title': getattr(result, 'title', ''),
                                'url': getattr(result, 'link', getattr(result, 'url', '')),
                                'source': getattr(result, 'source', 'keyword_search')
                            },
                            'score': 1.0 - (i * 0.05)  # Slight score decay
                        }
                    formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        min_relevance: float = 0.0,
        domains: Optional[List[str]] = None,
        use_hybrid: bool = True,
        **kwargs  # Accept additional kwargs for compatibility with hybrid_search
    ) -> List[Dict]:
        """
        Perform a hybrid search and return formatted results.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            min_relevance: Minimum relevance score (0.0-1.0)
            domains: Optional list of domains to filter results
            use_hybrid: Whether to use hybrid search. If False, uses semantic search only.
            **kwargs: Additional keyword arguments for compatibility
            
        Returns:
            List of search results with metadata
        """
        # Handle the 'k' parameter for compatibility with hybrid_search
        k = kwargs.get('k', num_results)
        
        logger.info(f"Performing search: {query}")
        
        try:
            results = []
            
            if use_hybrid and self.keyword_searcher:
                # Create a bound method with the instance
                bound_wrapper = lambda q, k, **kw: self._keyword_search_wrapper(q, k, **kw)
                bound_wrapper.search_async = self._keyword_search_wrapper
                
                try:
                    # Try hybrid search first
                    results = await hybrid_search(
                        semantic_searcher=self.semantic_searcher,
                        keyword_searcher=bound_wrapper,
                        query=query,
                        k=k * 2,  # Get extra results for filtering
                        alpha=0.7
                    )
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back to semantic search: {e}")
            
            # If hybrid search failed or wasn't requested, try semantic search only
            if not results:
                logger.info("Using semantic search only")
                try:
                    semantic_results = self.semantic_searcher.search(
                        query=query,
                        k=k * 2,
                        score_threshold=min_relevance
                    )
                    results = self._format_results(semantic_results)
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")
                    results = []
            
            # Format and filter results
            if results:
                formatted_results = self._format_results(results)
                filtered_results = self.filter_results(
                    formatted_results,
                    min_relevance=min_relevance,
                    domains=domains
                )[:num_results]  # Limit to requested number of results
            else:
                logger.warning("No results found from any search method")
                filtered_results = []
            
            # Add to search history
            search_entry = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_results': len(filtered_results),
                'results': filtered_results,
                'search_type': 'hybrid' if use_hybrid and self.keyword_searcher else 'semantic'
            }
            self.search_history.append(search_entry)
            
            # Save updated history
            self._save_data()
            
            # Print results to terminal in a formatted way
            self._print_results(query, filtered_results)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty list on error
            return []
    
    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results for display and export."""
        formatted = []
        
        for i, result in enumerate(results, 1):
            # Extract relevant fields
            metadata = result.get('metadata', {})
            
            formatted_result = {
                'rank': i,
                'title': metadata.get('title', ''),
                'url': metadata.get('url', ''),
                'snippet': result.get('text', '')[:200] + '...',
                'relevance_score': float(result.get('score', 0.0)),
                'source': metadata.get('source', 'unknown'),
                'domain': self._extract_domain(metadata.get('url', '')),
                'metadata': metadata  # Keep full metadata for reference
            }
            
            formatted.append(formatted_result)
        
        return formatted
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return 'N/A'
        try:
            # Remove protocol and www. if present
            domain = url.split('//')[-1].split('/')[0]
            domain = domain.replace('www.', '')
            # Take only the main domain (e.g., 'example.com' from 'sub.example.com')
            return '.'.join(domain.split('.')[-2:])
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {e}")
        return 'N/A'
    
    def filter_results(
        self,
        results: List[Dict],
        min_relevance: float = 0.0,
        domains: Optional[List[str]] = None,
        max_length: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter search results based on relevance and domain.
        
        Args:
            results: List of search results
            min_relevance: Minimum relevance score (0.0-1.0)
            domains: List of allowed domains (None for all)
            max_length: Maximum number of results to return
            
        Returns:
            Filtered list of results
        """
        filtered = [
            r for r in results 
            if r['relevance_score'] >= min_relevance
        ]
        
        if domains:
            filtered = [
                r for r in filtered 
                if any(d.lower() in r['url'].lower() for d in domains)
            ]
        
        # Sort by relevance score (descending)
        filtered.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(filtered, 1):
            result['rank'] = i
        
        return filtered[:max_length] if max_length else filtered
    
    def _print_results(self, query: str, results: List[Dict]) -> None:
        """
        Print search results in a formatted way to the terminal.
        
        Args:
            query: The search query
            results: List of search results to display
        """
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        if not results:
            console.print("[yellow]No results found.[/]")
            return
        
        # Create a table for the results
        table = Table(title=f"Search Results for: '{query}'")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Relevance", style="green")
        table.add_column("Domain", style="blue")
        
        # Add rows to the table
        for result in results:
            title = result.get('title', 'No title')
            if len(title) > 50:
                title = title[:47] + '...'
                
            table.add_row(
                str(result.get('rank', '')),
                title,
                f"{result.get('relevance_score', 0):.2f}",
                result.get('domain', 'N/A')
            )
        
        # Print the table
        console.print(table)
        
        # Print snippets for each result
        for i, result in enumerate(results, 1):
            console.print(f"\n[bold]{i}. {result.get('title', 'No title')}[/]")
            console.print(f"[blue]URL:[/] {result.get('url', 'N/A')}")
            console.print(f"[green]Relevance: {result.get('relevance_score', 0):.2f}")
            console.print(f"\n{result.get('snippet', 'No snippet available')}")
            console.print("-" * 80)
    
    def add_training_example(
        self,
        query: str,
        preferred_results: List[Dict],
        negative_results: Optional[List[Dict]] = None
    ) -> None:
        """
        Add a training example to improve search quality.
        
        Args:
            query: The search query
            preferred_results: List of preferred search results
            negative_results: Optional list of negative examples
        """
        example = {
            'query': query,
            'preferred_results': preferred_results,
            'negative_results': negative_results or [],
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data.append(example)
        self._save_data()
        logger.info(f"Added training example for query: {query}")
    
    def fine_tune_model(
        self,
        output_path: Optional[str] = None,
        batch_size: int = 16,
        epochs: int = 3,
        warmup_steps: int = 100,
        show_progress_bar: bool = True
    ) -> str:
        """
        Fine-tune the semantic search model on collected training data.
        
        Args:
            output_path: Directory to save the fine-tuned model
            batch_size: Training batch size
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            show_progress_bar: Whether to show training progress
            
        Returns:
            Path to the fine-tuned model
        """
        if not self.training_data:
            raise ValueError("No training data available. Add examples with add_training_example()")
        
        # Convert training data to the format expected by sentence-transformers
        train_examples = []
        for example in self.training_data:
            query = example['query']
            
            # Add positive examples
            for pos in example.get('preferred_results', []):
                train_examples.append({
                    'query': query,
                    'pos': pos.get('text', pos.get('snippet', ''))
                })
            
            # Add negative examples if available
            for neg in example.get('negative_results', []):
                if train_examples:  # Only add if we have positive examples
                    train_examples[-1]['neg'] = neg.get('text', neg.get('snippet', ''))
        
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.data_dir / f"fine_tuned_model_{timestamp}"
        
        # Fine-tune the model
        self.semantic_searcher.fine_tune(
            train_examples=train_examples,
            output_path=output_path,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=show_progress_bar
        )
        
        # Update the model in the semantic searcher
        self.semantic_searcher = SemanticSearchTool(
            model_name=output_path,
            use_gpu=self.use_gpu
        )
        
        logger.info(f"Model fine-tuned and saved to {output_path}")
        return output_path
    
    def get_search_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get search history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of search history entries
        """
        if limit:
            return self.search_history[-limit:]
        return self.search_history
    
    def clear_history(self) -> None:
        """Clear search history."""
        self.search_history = []
        self._save_data()
        logger.info("Search history cleared")

async def _search_social_media(self, query: str, **kwargs) -> List[Dict]:
    """Search across all configured social media sites."""
    all_results = []
    
    for site in self.SOCIAL_MEDIA_SITES:
        site_query = f"{query} {site}"
        try:
            if hasattr(self.keyword_searcher, 'search_async'):
                results = await self.keyword_searcher.search_async(
                    site_query, 
                    max_results=self.RESULTS_PER_SITE,
                    **kwargs
                )
            else:
                results = self.keyword_searcher.search(
                    site_query,
                    max_results=self.RESULTS_PER_SITE,
                    **kwargs
                )
            
            if results:
                # Add source information
                for result in results:
                    if not isinstance(result, dict):
                        result = result.__dict__
                    result['source_site'] = site.replace('site:', '')
                all_results.extend(results)
                
        except Exception as e:
            logger.warning(f"Search failed for {site}: {e}")
            continue
    
    return all_results

def _format_search_results(self, results: List[Union[Dict, Any]]) -> List[Dict]:
    """Format and deduplicate search results."""
    formatted = []
    seen_urls = set()
    
    for result in results:
        try:
            # Convert to dict if it's an object
            if not isinstance(result, dict):
                result = result.__dict__
                
            url = result.get('url') or result.get('link')
            if not url or url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            formatted.append({
                'text': result.get('snippet', result.get('description', '')),
                'metadata': {
                    'title': result.get('title', 'No title'),
                    'url': url,
                    'source': result.get('source_site', 'web'),
                    'source_type': 'social' if result.get('source_site') else 'web'
                },
                'score': 1.0  # Will be adjusted during ranking
            })
        except Exception as e:
            logger.warning(f"Error formatting result: {e}")
    
    return formatted

async def _keyword_search_wrapper(self, query: str, k: int = 10, **kwargs) -> List[Dict]:
    """Wrapper method to perform searches across multiple sources."""
    all_results = []
    
    # 1. Regular search
    try:
        if hasattr(self.keyword_searcher, 'search_async'):
            results = await self.keyword_searcher.search_async(query, max_results=k, **kwargs)
        else:
            results = self.keyword_searcher.search(query, max_results=k, **kwargs)
        all_results.extend(results or [])
    except Exception as e:
        logger.warning(f"Regular search failed: {e}")
    
    # 2. Social media searches
    try:
        social_results = await self._search_social_media(query, **kwargs)
        all_results.extend(social_results)
    except Exception as e:
        logger.warning(f"Social media search failed: {e}")
    
    # Format and return results
    return self._format_search_results(all_results)

def example_usage():
    """
    Example usage of the SearchAssistant class.
    
    This function demonstrates the main features of the SearchAssistant,
    including searching, filtering, and improving results through feedback.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    console.print(Panel.fit("SEARCH ASSISTANT DEMONSTRATION", style="bold blue"))
    
    # 1. Initialize the search assistant
    console.print("\n[bold]1. Initializing SearchAssistant...[/]")
    assistant = SearchAssistant(
        model_name="all-mpnet-base-v2",  # Pre-trained model
        data_dir="./data/search_assistant",  # Where to store data
        use_gpu=False  # Set to True if you have a GPU
    )
    
    # 2. Perform a basic search
    console.print("\n[bold]2. Performing a basic search...[/]")
    query = "the rock diet and workout"  ##Type in Search Query Here!!!###
    console.print(f"Searching for: [cyan]{query}[/]")
    
    with console.status("[bold green]Searching..."):
        results = assistant.search(
            query=query,
            num_results=5, #Max search results
            min_relevance=0.2,  # Only include results with at least 20% relevance
            domains=["site:youtube.com", "site:instagram.com", "site:twitter.com", "site:facebook.com", "site:tiktok.com", "site:reddit.com"]  # Filter by domain
        )
    
    # 3. Add training examples to improve results
    console.print("\n[bold]3. Adding training examples to improve search quality...[/]")
    training_text = Text()
    training_text.append("Adding training examples for better ML library results\n")
    training_text.append("• Positive: ", style="green")
    training_text.append("TensorFlow, PyTorch\n")
    training_text.append("• Negative: ", style="red")
    training_text.append("General programming articles")
    
    console.print(Panel(training_text, title="Training Data", border_style="blue"))
    
    assistant.add_training_example(
        query="the rock workout and diet",
        preferred_results=[
            {"text": "The Rock's workout routine."},
            {"text": "The Rock's diet."}
        ],
        negative_results=[
            {"text": "General fitness articles."}
        ]
    )
    
    console.print("[green]✓ Training examples added successfully![/]")
    
    # 4. View search history
    console.print("\n[bold]4. Viewing search history:[/]")
    history = assistant.get_search_history(limit=3)
    
    if history:
        for i, entry in enumerate(history, 1):
            query = entry.get('query', 'No query')
            num_results = entry.get('num_results', 0)
            console.print(f"{i}. [cyan]{query[:60]}[/]... ([green]{num_results} results[/])")
    else:
        console.print("[yellow]No search history found.[/]")
    
    # 5. Show usage example
    console.print("\n[bold]5. How to use in your code:[/]")
    usage_code = """
# Import the SearchAssistant
from scraper_llm.search_assistant import SearchAssistant

# Initialize the assistant
assistant = SearchAssistant()

# Perform a search
results = assistant.search(
    "your search query",
    num_results=5,
    min_relevance=0.3,
    domains=["example.com"]  # Optional domain filter
)

# Results are automatically printed in a formatted way
# You can also access the raw results:
for result in results:
    print(f"{result['title']} ({result['relevance_score']:.2f})")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")
    print("-" * 80)
"""
    console.print(Panel(usage_code, title="Code Example", border_style="green"))
    
    console.print("\n[bold green]✓ DEMONSTRATION COMPLETE[/]")
    console.print("=" * 80)


if __name__ == "__main__":
    example_usage()
