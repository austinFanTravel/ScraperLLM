"""
Semantic Search Utilities for ScraperLLM

This module provides utilities for semantic search functionality including:
- Document indexing and searching
- Model fine-tuning
- Evaluation metrics
- Integration with existing search pipelines
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
from tqdm import tqdm

# Import sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Some features will be disabled.")

# Import Annoy if available
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    logger.warning("annoy not available. Vector search will be disabled.")


class SemanticSearchTool:
    """
    A comprehensive semantic search tool that supports indexing, searching, and fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        index_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the semantic search tool.
        
        Args:
            model_name: Name of the sentence transformer model to use
            index_path: Path to load a pre-built index
            use_gpu: Whether to use GPU if available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for SemanticSearchTool")
            
        if not ANNOY_AVAILABLE:
            logger.warning("Annoy is not available. Falling back to exact search which may be slower.")
            
        self.model = SentenceTransformer(model_name)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            self.model = self.model.cuda()
            
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.documents = []
        self.metadatas = []
        
        # Initialize Annoy index if available
        if ANNOY_AVAILABLE:
            self.index = AnnoyIndex(self.embedding_dim, 'angular')
            self.index_built = False
        else:
            self.index = None
            self.embeddings = []
            
        # Load index if path is provided
        if index_path:
            self.load_index(index_path)
    
    def add_documents(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32
    ) -> None:
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents (either strings or dictionaries with 'text' key)
            metadatas: Optional list of metadata dictionaries
            batch_size: Batch size for embedding generation
        """
        # Convert documents to text and extract metadata
        texts = []
        new_metadatas = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                texts.append(doc)
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                new_metadatas.append(metadata)
            elif isinstance(doc, dict) and 'text' in doc:
                texts.append(doc['text'])
                metadata = {k: v for k, v in doc.items() if k != 'text'}
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])
                new_metadatas.append(metadata)
            else:
                raise ValueError(f"Document {i} must be a string or dictionary with 'text' key")
        
        # Generate embeddings in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing documents"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to index
            for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                idx = len(self.documents)
                self.documents.append(text)
                self.metadatas.append(new_metadatas[i + j])
                
                if self.index is not None:
                    self.index.add_item(idx, embedding)
                else:
                    self.embeddings.append(embedding)
    
    def build_index(self, n_trees: int = 10) -> None:
        """
        Build the search index for efficient similarity search.
        
        Args:
            n_trees: Number of trees for the Annoy index (more trees = more accurate but slower)
        """
        if not self.documents:
            raise ValueError("No documents have been added to the index")
            
        if self.index is not None:
            logger.info(f"Building index with {n_trees} trees...")
            self.index.build(n_trees)
            self.index_built = True
        else:
            logger.info("Using exact search (no Annoy index)")
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of dictionaries containing document text, metadata, and score
        """
        if not self.documents:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        # Perform search
        if self.index is not None and hasattr(self, 'index_built') and self.index_built:
            # Use Annoy for approximate nearest neighbor search
            indices = self.index.get_nns_by_vector(
                query_embedding, 
                n=k, 
                search_k=-1,  # Search all nodes
                include_distances=True
            )
            
            # Convert distances to similarity scores (1 - normalized distance)
            results = []
            for idx, distance in zip(*indices):
                # Angular distance is in [0, 2], convert to similarity [0, 1]
                score = 1 - (distance / 2)
                if score >= score_threshold:
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score)
                    })
        else:
            # Fall back to exact search (slower for large datasets)
            if not hasattr(self, 'embeddings') or not self.embeddings:
                self.embeddings = self.model.encode(
                    self.documents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            
            # Calculate cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            doc_embeddings = np.array(self.embeddings)
            doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # Get top-k results
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= score_threshold:
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': score
                    })
        
        return results
    
    def save_index(self, dir_path: str) -> None:
        """
        Save the search index and document data to disk.
        
        Args:
            dir_path: Directory to save the index files
        """
        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, "semantic_index")
        
        # Save document data
        with open(f"{base_path}_docs.json", 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'model_name': self.model_name if hasattr(self, 'model_name') else 'all-mpnet-base-v2'
            }, f, ensure_ascii=False, indent=2)
        
        # Save Annoy index if available
        if self.index is not None and hasattr(self, 'index_built') and self.index_built:
            self.index.save(f"{base_path}.ann")
        
        logger.info(f"Saved index to {base_path}.*")
    
    @classmethod
    def load_index(cls, dir_path: str, use_gpu: bool = False) -> 'SemanticSearchTool':
        """
        Load a saved search index from disk.
        
        Args:
            dir_path: Directory containing the saved index files
            use_gpu: Whether to use GPU if available
            
        Returns:
            An instance of SemanticSearchTool with the loaded index
        """
        base_path = os.path.join(dir_path, "semantic_index")
        
        # Load document data
        with open(f"{base_path}_docs.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize searcher
        model_name = data.get('model_name', 'all-mpnet-base-v2')
        searcher = cls(model_name=model_name, use_gpu=use_gpu)
        searcher.documents = data['documents']
        searcher.metadatas = data['metadatas']
        
        # Load Annoy index if available
        if os.path.exists(f"{base_path}.ann"):
            if not ANNOY_AVAILABLE:
                logger.warning("Annoy index found but Annoy is not available. Index will not be loaded.")
            else:
                searcher.index = AnnoyIndex(searcher.embedding_dim, 'angular')
                searcher.index.load(f"{base_path}.ann")
                searcher.index_built = True
        
        logger.info(f"Loaded index with {len(searcher.documents)} documents")
        return searcher
    
    def fine_tune(
        self,
        train_examples: List[Dict[str, str]],
        output_path: str,
        batch_size: int = 16,
        epochs: int = 3,
        warmup_steps: int = 100,
        evaluation_steps: int = 100,
        show_progress_bar: bool = True
    ) -> None:
        """
        Fine-tune the sentence transformer model on custom data.
        
        Args:
            train_examples: List of training examples as dictionaries with 'query', 'pos', and optionally 'neg' keys
            output_path: Directory to save the fine-tuned model
            batch_size: Training batch size
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduling
            evaluation_steps: Number of steps between evaluations (set to 0 to disable)
            show_progress_bar: Whether to show the progress bar during training
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for fine-tuning")
        
        # Convert examples to InputExample objects
        train_samples = []
        for example in train_examples:
            if 'neg' in example:
                train_samples.append(InputExample(
                    texts=[example['query'], example['pos'], example['neg']],
                    label=1.0  # Not used for MultipleNegativesRankingLoss
                ))
            else:
                train_samples.append(InputExample(
                    texts=[example['query'], example['pos']],
                    label=1.0
                ))
        
        # Create data loader
        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Define loss function
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        
        # Configure evaluator if evaluation steps > 0
        evaluator = None
        if evaluation_steps > 0 and len(train_examples) > 10:  # Only if we have enough examples
            # Create a small evaluation set from the training data
            eval_examples = train_examples[:10]  # Use first 10 examples for evaluation
            queries = [ex['query'] for ex in eval_examples]
            corpus = list({ex['pos'] for ex in eval_examples} | 
                        {ex.get('neg', '') for ex in eval_examples if 'neg' in ex})
            
            # Remove empty strings from corpus
            corpus = [doc for doc in corpus if doc.strip()]
            
            # Create evaluator
            evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                show_progress_bar=show_progress_bar
            )
        
        # Fine-tune the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            output_path=output_path,
            show_progress_bar=show_progress_bar
        )
        
        # Reload the fine-tuned model
        self.model = SentenceTransformer(output_path)
        if self.use_gpu:
            self.model = self.model.cuda()
        
        logger.info(f"Fine-tuned model saved to {output_path}")


def evaluate_search_quality(
    searcher: SemanticSearchTool,
    test_queries: List[Dict[str, Any]],
    k: int = 5,
    score_threshold: float = 0.0
) -> Dict[str, float]:
    """
    Evaluate the quality of search results.
    
    Args:
        searcher: Initialized SemanticSearchTool instance
        test_queries: List of test queries with expected results
                     Each item should be a dict with 'query' and 'expected' keys
        k: Number of results to consider for each query
        score_threshold: Minimum score threshold for considering a result relevant
        
    Returns:
        Dictionary containing evaluation metrics (precision@k, recall@k, mrr@k)
    """
    precisions = []
    recalls = []
    reciprocal_ranks = []
    
    for query_data in test_queries:
        query = query_data['query']
        expected = set(query_data['expected'])
        
        # Get search results
        results = searcher.search(query, k=k, score_threshold=score_threshold)
        
        # Extract retrieved documents
        retrieved = {result['text'] for result in results}
        
        # Calculate precision and recall
        if retrieved:
            true_positives = len(retrieved & expected)
            precision = true_positives / len(retrieved)
            recall = true_positives / len(expected) if expected else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate reciprocal rank
            for i, result in enumerate(results, 1):
                if result['text'] in expected:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
    
    # Calculate metrics
    metrics = {
        'precision@k': np.mean(precisions) if precisions else 0.0,
        'recall@k': np.mean(recalls) if recalls else 0.0,
        'mrr@k': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        'num_queries': len(test_queries)
    }
    
    return metrics


def hybrid_search(
    semantic_searcher: SemanticSearchTool,
    keyword_searcher: Any,  # Any search object with a search() method
    query: str,
    k: int = 5,
    alpha: float = 0.7,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Combine semantic and keyword search results.
    
    Args:
        semantic_searcher: Initialized SemanticSearchTool instance
        keyword_searcher: Any search object with a search(query, k) method
        query: Search query
        k: Number of results to return
        alpha: Weight for semantic search (1.0 = only semantic, 0.0 = only keyword)
        score_threshold: Minimum score threshold for semantic search results
        
    Returns:
        List of search results with combined scores
    """
    # Get semantic search results
    semantic_results = semantic_searcher.search(query, k=k * 2, score_threshold=score_threshold)
    
    # Get keyword search results
    try:
        keyword_results = keyword_searcher.search(query, k=k * 2)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return semantic_results[:k]
    
    # Combine results
    combined = {}
    
    # Add semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result.get('metadata', {}).get('id', result['text'])
        combined[doc_id] = {
            'text': result['text'],
            'metadata': result.get('metadata', {}),
            'semantic_score': result['score'],
            'keyword_score': 0.0,
            'combined_score': result['score'] * alpha
        }
    
    # Add keyword results
    for i, result in enumerate(keyword_results):
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        
        doc_id = result.get('metadata', {}).get('id', result.get('text', str(i)))
        score = 1.0 - (i / len(keyword_results))  # Normalize score based on rank
        
        if doc_id in combined:
            # Document exists in both, update scores
            combined[doc_id]['keyword_score'] = score
            combined[doc_id]['combined_score'] = (
                alpha * combined[doc_id]['semantic_score'] + 
                (1 - alpha) * score
            )
        else:
            # New document from keyword search
            combined[doc_id] = {
                'text': result.get('text', ''),
                'metadata': result.get('metadata', {}),
                'semantic_score': 0.0,
                'keyword_score': score,
                'combined_score': score * (1 - alpha)
            }
    
    # Sort by combined score and return top-k
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    return [
        {
            'text': r['text'],
            'metadata': r['metadata'],
            'score': r['combined_score'],
            'semantic_score': r['semantic_score'],
            'keyword_score': r['keyword_score']
        }
        for r in sorted_results[:k]
    ]
