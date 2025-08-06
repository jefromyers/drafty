"""Embeddings service for semantic search and content similarity."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingsService:
    """Service for creating and managing text embeddings."""
    
    # Recommended models for different use cases
    MODELS = {
        "default": "all-MiniLM-L6-v2",  # Fast, good quality
        "quality": "all-mpnet-base-v2",  # Higher quality, slower
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "fast": "all-MiniLM-L6-v2",
        "long": "allenai-specter",  # For scientific/long documents
    }
    
    def __init__(
        self,
        model_name: str = "default",
        cache_dir: Optional[Path] = None,
        device: str = "cpu"
    ):
        """Initialize embeddings service.
        
        Args:
            model_name: Model name or key from MODELS dict
            cache_dir: Directory for caching embeddings
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        # Get actual model name
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]
        
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".drafty" / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Cache for current session
        self._session_cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        # Check session cache first
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)
                    self._session_cache[cache_key] = embedding
                    return embedding
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        # Save to session cache
        self._session_cache[cache_key] = embedding
        
        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception:
            pass
    
    def embed_text(self, text: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """Create embeddings for text.
        
        Args:
            text: Single text or list of texts
            use_cache: Whether to use caching
        
        Returns:
            Embedding vector(s) as numpy array
        """
        # Handle single text
        if isinstance(text, str):
            if use_cache:
                cache_key = self._get_cache_key(text)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    return cached
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            if use_cache:
                self._save_to_cache(cache_key, embedding)
            
            return embedding
        
        # Handle batch
        embeddings = []
        texts_to_encode = []
        cache_keys = []
        cached_indices = []
        
        for i, t in enumerate(text):
            if use_cache:
                cache_key = self._get_cache_key(t)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    cached_indices.append(i)
                else:
                    texts_to_encode.append(t)
                    cache_keys.append(cache_key)
            else:
                texts_to_encode.append(t)
        
        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, convert_to_numpy=True)
            
            # Save to cache
            if use_cache:
                for key, emb in zip(cache_keys, new_embeddings):
                    self._save_to_cache(key, emb)
            
            # Merge with cached embeddings
            if cached_indices:
                result = np.zeros((len(text), self.embedding_dim))
                
                # Fill in cached embeddings
                for i, emb in zip(cached_indices, embeddings[:len(cached_indices)]):
                    result[i] = emb
                
                # Fill in new embeddings
                new_idx = 0
                for i in range(len(text)):
                    if i not in cached_indices:
                        result[i] = new_embeddings[new_idx]
                        new_idx += 1
                
                return result
            else:
                return new_embeddings
        else:
            return np.array(embeddings)
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "content",
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Embed a list of text chunks with metadata.
        
        Args:
            chunks: List of dicts with text and metadata
            text_field: Field name containing text to embed
            batch_size: Batch size for encoding
        
        Returns:
            Chunks with added 'embedding' field
        """
        # Extract texts
        texts = [chunk[text_field] for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)
        
        # Concatenate embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([])
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks
    
    def semantic_search(
        self,
        query: Union[str, np.ndarray],
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Find most similar items using semantic search.
        
        Args:
            query: Query text or embedding
            corpus_embeddings: Embeddings to search through
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (index, similarity_score) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embed_text(query)
        else:
            query_embedding = query
        
        # Ensure 2D arrays
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if corpus_embeddings.ndim == 1:
            corpus_embeddings = corpus_embeddings.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and return with scores
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((int(idx), score))
        
        return results
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 2,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Cluster embeddings using DBSCAN.
        
        Args:
            embeddings: Embeddings to cluster
            eps: Maximum distance between samples in same cluster
            min_samples: Minimum samples in a cluster
            metric: Distance metric
        
        Returns:
            Cluster labels (-1 for noise)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clustering.fit_predict(embeddings)
        
        return labels
    
    def find_diverse_items(
        self,
        query: Union[str, np.ndarray],
        corpus_embeddings: np.ndarray,
        corpus_metadata: Optional[List[Dict]] = None,
        top_k: int = 10,
        diversity_threshold: float = 0.3,
        relevance_weight: float = 0.7
    ) -> List[Tuple[int, float, int]]:
        """Find relevant but diverse items using clustering.
        
        Args:
            query: Query text or embedding
            corpus_embeddings: Embeddings to search
            corpus_metadata: Optional metadata for items
            top_k: Number of results
            diversity_threshold: Clustering threshold
            relevance_weight: Weight for relevance vs diversity
        
        Returns:
            List of (index, score, cluster_id) tuples
        """
        # Get initial candidates (3x what we need)
        candidates = self.semantic_search(
            query,
            corpus_embeddings,
            top_k=min(top_k * 3, len(corpus_embeddings))
        )
        
        if not candidates:
            return []
        
        # Extract candidate embeddings
        candidate_indices = [idx for idx, _ in candidates]
        candidate_embeddings = corpus_embeddings[candidate_indices]
        
        # Cluster candidates
        if len(candidate_indices) > 1:
            clusters = self.cluster_embeddings(
                candidate_embeddings,
                eps=diversity_threshold
            )
        else:
            clusters = np.array([0])
        
        # Select diverse items
        selected = []
        used_clusters = set()
        
        # First, get best item from each cluster
        for idx, score in candidates:
            cluster_id = clusters[candidate_indices.index(idx)]
            
            if cluster_id not in used_clusters or cluster_id == -1:
                selected.append((idx, score, int(cluster_id)))
                used_clusters.add(cluster_id)
                
                if len(selected) >= top_k:
                    break
        
        # If we need more, add remaining by score
        if len(selected) < top_k:
            for idx, score in candidates:
                if not any(s[0] == idx for s in selected):
                    cluster_id = clusters[candidate_indices.index(idx)]
                    selected.append((idx, score, int(cluster_id)))
                    
                    if len(selected) >= top_k:
                        break
        
        return selected[:top_k]
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        return float(cosine_similarity(embedding1, embedding2)[0, 0])
    
    def find_optimal_insertion_points(
        self,
        article_sections: List[str],
        link_content: str,
        max_points: int = 3,
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Find best places to insert a link in an article.
        
        Args:
            article_sections: List of article sections/paragraphs
            link_content: Content/description of link
            max_points: Maximum insertion points to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (section_index, similarity_score) tuples
        """
        # Embed link content
        link_embedding = self.embed_text(link_content)
        
        # Embed all sections
        section_embeddings = self.embed_text(article_sections)
        
        # Find similar sections
        results = self.semantic_search(
            link_embedding,
            section_embeddings,
            top_k=max_points,
            threshold=min_similarity
        )
        
        return results
    
    def group_by_topic(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        similarity_threshold: float = 0.3
    ) -> Dict[int, List[int]]:
        """Group texts by topic similarity.
        
        Args:
            texts: List of texts to group
            metadata: Optional metadata for texts
            similarity_threshold: Threshold for grouping
        
        Returns:
            Dict mapping cluster_id to list of text indices
        """
        # Embed texts
        embeddings = self.embed_text(texts)
        
        # Cluster
        labels = self.cluster_embeddings(
            embeddings,
            eps=similarity_threshold
        )
        
        # Group by cluster
        groups = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)
        
        return groups
    
    def save_index(self, embeddings: np.ndarray, metadata: List[Dict], path: Path) -> None:
        """Save embeddings index to disk.
        
        Args:
            embeddings: Embedding vectors
            metadata: Associated metadata
            path: Path to save index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            "model_name": self.model_name,
            "embeddings": embeddings,
            "metadata": metadata,
            "embedding_dim": self.embedding_dim
        }
        
        with open(path, "wb") as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: Path) -> Tuple[np.ndarray, List[Dict]]:
        """Load embeddings index from disk.
        
        Args:
            path: Path to index file
        
        Returns:
            Tuple of (embeddings, metadata)
        """
        with open(path, "rb") as f:
            index_data = pickle.load(f)
        
        # Verify model compatibility
        if index_data.get("model_name") != self.model_name:
            print(f"Warning: Index was created with {index_data.get('model_name')}, "
                  f"but current model is {self.model_name}")
        
        return index_data["embeddings"], index_data["metadata"]