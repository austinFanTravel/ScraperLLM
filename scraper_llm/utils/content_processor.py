"""Content processing and summarization utilities for ScraperLLM."""
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import aiohttp
import re
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
from bs4 import BeautifulSoup
from loguru import logger
import re

@dataclass
class WebContent:
    """Container for processed web content."""
    url: str
    title: str
    text: str
    relevance: float = 0.0

class ContentProcessor:
    """Handles web content extraction and summarization."""
    
    def __init__(self, max_content_length: int = 10000, use_llm: bool = True):
        """Initialize the content processor.
        
        Args:
            max_content_length: Maximum number of characters to process per page
            use_llm: Whether to use local LLM for answer generation
        """
        self.max_content_length = max_content_length
        self.session = aiohttp.ClientSession()
        self.llm = None
        self.use_llm = use_llm
        
        if use_llm:
            try:
                from .llm_integration import LocalLLM
                self.llm = LocalLLM()
            except ImportError:
                logger.warning("llama-cpp-python not installed. LLM features will be disabled.")
                self.use_llm = False
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.use_llm = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP session."""
        if hasattr(self, 'session') and not self.session.closed:
            await self.session.close()
    
    async def extract_content(self, url: str, verify_ssl: bool = True) -> Optional[WebContent]:
        """Extract main content from a web page.
        
        Args:
            url: URL of the page to extract content from
            verify_ssl: Whether to verify SSL certificates. Set to False to skip verification (less secure)
            
        Returns:
            WebContent object with extracted content, or None if extraction fails
        """
        import ssl
        import certifi
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        if not verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            async with self.session.get(
                url,
                ssl=ssl_context,
                headers=headers,
                timeout=30,
                allow_redirects=True
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return None

                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"Skipping non-HTML content at {url} (Content-Type: {content_type})")
                    return None

                html = await response.text()
                
                # Try different parsers in order of preference
                parsers = ['lxml', 'html.parser', 'html5lib']
                soup = None
                last_error = None
                
                for parser in parsers:
                    try:
                        soup = BeautifulSoup(html, parser)
                        break  # Successfully parsed
                    except Exception as e:
                        last_error = e
                        logger.debug(f"Failed to parse with {parser}: {e}")
                        continue
                
                if soup is None:
                    logger.error(f"Could not parse HTML from {url} with any available parser. Last error: {last_error}")
                    return None
                
                # Extract title
                title = soup.title.string.strip() if soup.title else 'No title'
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'svg', 'button', 'form', 'aside']):
                    element.decompose()
                
                # Try to extract main content
                content_selectors = [
                    'article',
                    'main',
                    'div.article',
                    'div.content',
                    'div.post',
                    'div.entry-content',
                    'div#content',
                    'div#main',
                    'div#article',
                    'div#post',
                    'div.article-body',
                    'div.article-content',
                    'div.post-content',
                ]
                
                content = None
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        break
                
                # Fall back to body if no content found
                content = content or soup.body or soup
                
                # Extract text from content
                elements = content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
                text = '\n'.join(
                    element.get_text(' ', strip=True)
                    for element in elements
                    if element.get_text(strip=True)
                )
                
                # If no text was extracted, try getting all text
                if not text.strip():
                    text = content.get_text('\n', strip=True)
                
                # Clean up text
                text = '\n'.join(
                    ' '.join(line.split()) 
                    for line in text.splitlines()
                    if line.strip()
                )
                
                # Limit content length
                if len(text) > self.max_content_length:
                    text = text[:self.max_content_length] + '... [content truncated]'
                
                if text.strip():
                    return WebContent(url=url, title=title, text=text)
                else:
                    logger.warning(f"No content extracted from {url}")
                    return None
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}", exc_info=True)
            return None
    
    # Common restaurant-related terms that often appear in names
    RESTAURANT_TERMS = {
        'restaurant', 'cafe', 'bistro', 'eatery', 'grill', 'kitchen', 'bar', 'lounge', 
        'diner', 'brasserie', 'tavern', 'steakhouse', 'pizzeria', 'trattoria', 'ristorante', 
        'bakery', 'deli', 'pub', 'bistro', 'brasserie', 'chophouse', 'oyster bar', 'wine bar'
    }

    # Common words that are likely not restaurant names on their own
    COMMON_WORDS = {
        'the', 'and', 'or', 'for', 'with', 'that', 'this', 'from', 'have', 'they',
        'there', 'their', 'what', 'your', 'when', 'where', 'here', 'just', 'about', 'like'
    }
    
    def _is_likely_restaurant_name(self, name: str) -> bool:
        """Check if a name is likely a restaurant name."""
        if not name or len(name) < 3:  # Too short to be meaningful
            return False
            
        name_lower = name.lower()
        words = name_lower.split()
        
        # Check for common non-restaurant words
        if any(word in self.COMMON_WORDS for word in words):
            return False
            
        # Check for restaurant-related terms
        if any(term in name_lower for term in self.RESTAURANT_TERMS):
            return True
            
        # Check for common patterns in restaurant names
        if any(sep in name for sep in [' & ', ' and ', ' at ', ' of ', "'s "]):
            return True
            
        # Check title case (but not at sentence start)
        if (1 < len(words) <= 4 and  # Not too long, not a single word
            all(word[0].isupper() for word in name.split() if len(word) > 2)  # Title case
        ):
            return True
            
        return False

    def _extract_key_information(self, query: str, contents: List[WebContent]) -> Dict[str, List[str]]:
        """Extract key information from content based on the query."""
        info = {
            'locations': set(),
            'events': set(),
            'businesses': set(),
            'descriptions': []
        }
        
        query_lower = query.lower()
        
        for content in contents:
            if not content.text:
                continue
                
            text_lower = content.text.lower()
            
            # Extract business names (look for proper nouns near keywords)
            if any(word in query_lower for word in ['restaurant', 'bar', 'club', 'cafe', 'hotel', 'eatery']):
                # First, try to extract from numbered lists (like in PureWow article)
                numbered_list = re.findall(r'\d+\.\s*([^•\n]+?)(?=\s*\d+\.|$)', content.text)
                
                # Clean and validate the extracted names
                for item in numbered_list:
                    # Split on common separators and take the first part
                    name = re.split(r'[•·\-–—]', item)[0].strip()
                    if self._is_likely_restaurant_name(name):
                        info['businesses'].add(name)
                
                # If no valid names found in numbered list, try other patterns
                if not info['businesses']:
                    # Look for patterns like "• Restaurant Name" or "* Restaurant Name"
                    bullet_items = re.findall(r'[•*]\s*([A-Z][^•*\n]+?)(?=\s*[•*]|$)', content.text)
                    for item in bullet_items:
                        name = item.strip()
                        if self._is_likely_restaurant_name(name):
                            info['businesses'].add(name)
                
                # As a fallback, look for proper nouns near restaurant-related words
                if not info['businesses']:
                    sentences = content.text.split('.')
                    for sentence in sentences:
                        if any(word in sentence.lower() for word in ['restaurant', 'bar', 'cafe', 'eatery', 'diner']):
                            words = sentence.split()
                            for i, word in enumerate(words):
                                word = word.strip('"\'()[]{}')
                                if (word and word[0].isupper() and len(word) > 2 and 
                                    word.lower() not in self.COMMON_WORDS):
                                    # Take up to 3 words as the business name
                                    name = ' '.join(words[i:i+3])
                                    name = re.sub(r'[^\w\s]', '', name)  # Remove special chars
                                    if self._is_likely_restaurant_name(name):
                                        info['businesses'].add(name.strip())
                                        break
            
            # Extract locations
            if any(word in query_lower for word in ['where', 'location', 'address', 'in ', 'at ']):
                # Look for location indicators
                location_indicators = ['in ', 'at ', 'on ', 'street', 'avenue', 'ave', 'st', 'blvd']
                for indicator in location_indicators:
                    if indicator in text_lower:
                        # Get the words around the indicator as potential location
                        words = content.text.lower().split(indicator, 1)
                        if len(words) > 1:
                            location = words[1].split('.')[0].split('\n')[0].strip()
                            if location:
                                info['locations'].add(location.capitalize())
            
            # Extract events/dates
            if any(word in query_lower for word in ['when', 'date', 'time', 'event']):
                # Look for dates or times
                date_indicators = ['january', 'february', 'march', 'april', 'may', 'june', 
                                 'july', 'august', 'september', 'october', 'november', 'december',
                                 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                
                for indicator in date_indicators:
                    if indicator in text_lower:
                        # Get the sentence containing the date
                        for sentence in content.text.split('.'):
                            if indicator in sentence.lower():
                                info['events'].add(sentence.strip() + '.')
                                break
            
            # Store first 200 chars of content as description
            if content.text:
                info['descriptions'].append(content.text[:200].strip())
        
        # Filter out any remaining unwanted phrases
        filtered_businesses = []
        for business in info['businesses']:
            words = business.split()
            # Only include if it doesn't start with a verb or article
            if (words and words[0].lower() not in ['the', 'a', 'an', 'and', 'or', 'but'] and
                not any(word.lower() in self.COMMON_WORDS for word in words)):
                filtered_businesses.append(business)
        
        # Convert sets to lists and limit the number of items
        return {
            'locations': list(info['locations'])[:3],
            'events': list(info['events'])[:2],
            'businesses': filtered_businesses[:5],  # Limit to top 5 businesses
            'descriptions': info['descriptions'][:2]
        }
    
    def _generate_coherent_summary(self, query: str, info: Dict[str, List[str]]) -> str:
        """Generate a summary showing all answers from different sources."""
        query_lower = query.lower()
        answers = set()
        
        # Add business names if relevant to query
        if any(word in query_lower for word in ['restaurant', 'bar', 'club', 'cafe', 'hotel', 'place', 'location']):
            answers.update(info['businesses'])
        
        # Add locations if relevant
        if any(word in query_lower for word in ['where', 'location', 'address', 'in ', 'at ']):
            answers.update(info['locations'])
        
        # Add event information if relevant
        if any(word in query_lower for word in ['when', 'date', 'time', 'event']):
            answers.update(info['events'])
        
        # If we have specific answers, format them
        if answers:
            # Clean and deduplicate answers
            clean_answers = []
            for ans in answers:
                # Remove any trailing punctuation and add period
                clean = ans.strip().rstrip('.').strip()
                if clean and len(clean) > 2:  # Only include meaningful answers
                    clean_answers.append(clean)
            
            if clean_answers:
                # Format as "Query: answer1, answer2, answer3."
                return f"{query}: {', '.join(clean_answers[:5])}."
        
        # Fallback to descriptions if no specific answers found
        if info['descriptions']:
            # Take first sentence from each description
            desc_answers = []
            for desc in info['descriptions']:
                first_sent = desc.split('.')[0].strip()
                if first_sent and len(first_sent) > 10:  # Only include if meaningful
                    desc_answers.append(first_sent)
            
            if desc_answers:
                return f"{query}: {', '.join(desc_answers[:3])}."
        
        # Final fallback
        return f"Could not find specific information about {query}."
    
    async def generate_summary(self, query: str, contents: List[WebContent]) -> str:
        """Generate a concise summary based on the query and contents using LLM if available."""
        if not contents:
            return f"Could not find information about {query}."
        
        # Try using LLM for answer generation if available
        if self.use_llm and self.llm:
            try:
                # Prepare context from top results
                context = "\n\n".join(
                    f"Source {i+1}:\n{content.text[:2000]}"
                    for i, content in enumerate(contents[:3])  # Use top 3 results
                )
                
                # Generate answer using LLM
                answer = self.llm.answer_question(context, query)
                
                # Add source attribution
                sources = ', '.join(set(c.url for c in contents[:2]))
                return f"{answer} (Sources: {sources})"
                
            except Exception as e:
                logger.error(f"Error generating LLM answer: {e}")
                # Fall back to traditional method if LLM fails
        
        # Traditional method if LLM is not available or fails
        info = self._extract_key_information(query, contents)
        summary = self._generate_coherent_summary(query, info)
        sources = ', '.join(set(c.url for c in contents[:2]))  # Limit to 2 sources
        return f"{summary}\n\nSources: {sources}"
