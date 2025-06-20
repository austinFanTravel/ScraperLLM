"""Test script for LLM integration."""
import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scraper_llm.utils.content_processor import ContentProcessor, WebContent

async def test_llm_integration():
    """Test the LLM integration with a sample query and content."""
    print("Testing LLM integration...")
    
    # Create a sample WebContent object
    sample_content = WebContent(
        url="https://example.com/paris",
        title="Paris Travel Guide",
        text="""
        Paris is the capital of France and is known as the City of Light. 
        The Eiffel Tower, built by Gustave Eiffel, is one of the most 
        famous landmarks in the world. The city is also home to the 
        Louvre Museum, which houses the Mona Lisa.
        """
    )
    
    # Initialize the content processor with LLM support
    processor = ContentProcessor(use_llm=True)
    
    # Test a question
    question = "Who built the Eiffel Tower?"
    print(f"\nQuestion: {question}")
    
    # Generate answer
    answer = await processor.generate_summary(question, [sample_content])
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    asyncio.run(test_llm_integration())
