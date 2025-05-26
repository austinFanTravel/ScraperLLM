"""Named Entity Recognition utilities for ScraperLLM."""
import asyncio
from typing import Any, Dict, List, Optional, Union, cast

import nltk
from loguru import logger

from ..search.base import Entity, EntityType, SearchResult

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)

# Map NLTK entity types to our EntityType enum
NLTK_TO_ENTITY_TYPE = {
    "PERSON": EntityType.PERSON,
    "ORGANIZATION": EntityType.ORGANIZATION,
    "GPE": EntityType.LOCATION,
    "LOCATION": EntityType.LOCATION,
    "FACILITY": EntityType.FACILITY,
    "DATE": EntityType.DATE,
    "TIME": EntityType.TIME,
    "MONEY": EntityType.MONEY,
    "PERCENT": EntityType.PERCENT,
}

# Common entity patterns for post-processing
PERSON_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "madam", "lord", "lady"}
LOCATION_KEYWORDS = {"street", "avenue", "boulevard", "road", "lane", "drive", "st", "ave", "blvd", "rd", "ln", "dr"}


def extract_entities(
    texts: Union[str, List[str]],
    entity_types: Optional[List[EntityType]] = None
) -> Union[List[Entity], List[List[Entity]]]:
    """Extract named entities from text using NLTK.
    
    Args:
        texts: A single text string or a list of text strings to process
        entity_types: Optional list of entity types to include. If None, all types are included.
        
    Returns:
        If input is a single string: List of Entity objects
        If input is a list: List of lists of Entity objects
    """
    single_text = isinstance(texts, str)
    if single_text:
        texts = [cast(str, texts)]
    
    results = []
    for text in texts:
        if not text or not isinstance(text, str):
            results.append([])
            continue
            
        try:
            # Tokenize and tag text
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            
            # Chunk entities
            chunked = nltk.ne_chunk(tagged, binary=False)
            
            # Extract entities
            entities: List[Entity] = []
            
            for chunk in chunked:
                if hasattr(chunk, 'label'):
                    nltk_label = chunk.label()
                    entity_text = ' '.join(c[0] for c in chunk.leaves())
                    
                    # Map NLTK labels to our EntityType
                    entity_type = NLTK_TO_ENTITY_TYPE.get(nltk_label, EntityType.OTHER)
                    
                    # Skip if not in requested entity types
                    if entity_types and entity_type not in entity_types:
                        continue
                    
                    # Find the position in the original text
                    start_pos = text.find(entity_text)
                    if start_pos == -1:
                        continue  # Skip if not found (shouldn't happen with NLTK)
                        
                    entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(entity_text),
                    })
            
            # Post-process entities
            entities = _post_process_entities(text, entities)
            results.append(entities)
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            results.append([])
    
    return results[0] if single_text else results


async def extract_entities_async(
    texts: Union[str, List[str]],
    entity_types: Optional[List[EntityType]] = None
) -> Union[List[Entity], List[List[Entity]]]:
    """Asynchronously extract named entities from text.
    
    Args:
        texts: A single text string or a list of text strings to process
        entity_types: Optional list of entity types to include. If None, all types are included.
        
    Returns:
        If input is a single string: List of Entity objects
        If input is a list: List of lists of Entity objects
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: extract_entities(texts, entity_types=entity_types)
    )


def _post_process_entities(text: str, entities: List[Entity]) -> List[Entity]:
    """Post-process entities to improve quality.
    
    Args:
        text: The original text
        entities: List of extracted entities
        
    Returns:
        List of processed entities
    """
    processed = []
    seen = set()
    
    for entity in entities:
        # Skip duplicates
        key = (entity["text"].lower(), entity["type"])
        if key in seen:
            continue
        seen.add(key)
        
        # Clean up entity text
        entity_text = entity["text"].strip()
        if not entity_text:
            continue
            
        # Skip very short entities unless they're important
        if len(entity_text) < 2 and entity["type"] not in {
            EntityType.MONEY, EntityType.PERCENT, EntityType.TIME
        }:
            continue
            
        # Update positions based on cleaned text
        start_pos = text.find(entity_text, entity["start_pos"])
        if start_pos == -1:
            continue
            
        entity["text"] = entity_text
        entity["start_pos"] = start_pos
        entity["end_pos"] = start_pos + len(entity_text)
        
        processed.append(entity)
    
    # Sort by start position
    processed.sort(key=lambda x: x["start_pos"])
    return processed


def extract_entities_from_result(result: SearchResult) -> SearchResult:
    """Extract entities from a search result and add them to the result.
    
    Args:
        result: The search result to process
        
    Returns:
        The processed search result with entities
    """
    if not result.snippet:
        return result
        
    # Extract entities from title and snippet
    text = f"{result.title}. {result.snippet}"
    entities = extract_entities(text)
    
    # Add entities to the result
    for entity in entities:
        result.add_entity(
            text=entity["text"],
            entity_type=entity["type"],
            start_pos=entity["start_pos"],
            end_pos=entity["end_pos"]
        )
    
    return result
