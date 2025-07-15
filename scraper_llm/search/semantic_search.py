"""Semantic search functionality using Sentence Transformers and Annoy."""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import json

class SemanticSearcher:
    """A class for performing semantic search using Sentence Transformers and Annoy."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic searcher.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = AnnoyIndex(self.embedding_dim, 'angular')
        self.documents = []
        self.doc_metadata = []
        self.built = False
        
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the search index.
        
        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries for each document
        """
        start_idx = len(self.documents)
        
        # Generate embeddings for all texts
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Add to index
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            idx = start_idx + i
            self.index.add_item(idx, embedding)
            self.documents.append(text)
            
            # Store metadata if provided, otherwise use empty dict
            if metadatas and i < len(metadatas):
                self.doc_metadata.append(metadatas[i])
            else:
                self.doc_metadata.append({})
        
        logger.info(f"Added {len(texts)} documents to index")
    
    def build_index(self, n_trees: int = 10):
        """
        Build the Annoy index for fast similarity search.
        
        Args:
            n_trees: Number of trees to use in the index (more = more accurate but slower)
        """
        if not self.documents:
            raise ValueError("No documents have been added to the index")
            
        logger.info(f"Building index with {n_trees} trees...")
        self.index.build(n_trees)
        self.built = True
        logger.info("Index built successfully")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing 'text', 'metadata', and 'score' for each result
        """
        if not self.built:
            self.build_index()
            
        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        # Search the index
        indices = self.index.get_nns_by_vector(
            query_embedding, 
            n=k, 
            search_k=-1,  # Search all nodes
            include_distances=True
        )
        
        # Convert distances to similarity scores (1 - normalized distance)
        results = []
        for idx, distance in zip(*indices):
            # Annoy uses angular distance, convert to similarity score (1 - normalized distance)
            similarity = 1 - (distance / 2)  # Angular distance is in [0, 2]
            
            results.append({
                'text': self.documents[idx],
                'metadata': self.doc_metadata[idx],
                'score': float(similarity)
            })
        
        return results
    
    def save_index(self, dir_path: str = "./data/indices"):
        """
        Save the index and document data to disk.
        
        Args:
            dir_path: Directory to save the index files
        """
        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, f"semantic_index_{self.model_name.replace('/', '_')}")
        
        # Save Annoy index
        self.index.save(f"{base_path}.ann")
        
        # Save document data
        with open(f"{base_path}_docs.json", 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.doc_metadata,
                'model_name': self.model_name
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved index to {base_path}.ann and document data to {base_path}_docs.json")
    
    @classmethod
    def load_index(cls, dir_path: str = "./data/indices", model_name: str = 'all-MiniLM-L6-v2'):
        """
        Load a saved index from disk.
        
        Args:
            dir_path: Directory containing the saved index files
            model_name: Name of the Sentence Transformer model used to create the index
            
        Returns:
            An instance of SemanticSearcher with the loaded index
        """
        base_path = os.path.join(dir_path, f"semantic_index_{model_name.replace('/', '_')}")
        
        # Initialize searcher
        searcher = cls(model_name)
        
        # Load document data
        with open(f"{base_path}_docs.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            searcher.documents = data['documents']
            searcher.doc_metadata = data['metadata']
        
        # Load Annoy index
        searcher.index = AnnoyIndex(searcher.embedding_dim, 'angular')
        searcher.index.load(f"{base_path}.ann")
        searcher.built = True
        
        logger.info(f"Loaded index with {len(searcher.documents)} documents")
        return searcher
