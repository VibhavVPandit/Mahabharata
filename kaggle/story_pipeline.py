"""Story generation pipeline adapted for Kaggle environment."""
import os
import json
import yaml
import math
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

# Import adapted components
import sys
# Add parent directory to path to import from src if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Passage:
    """Represents a passage with its embedding and metadata."""
    chunk_id: str
    text: str
    embedding: np.ndarray
    characters: List[str]
    themes: List[str]
    chapter: Optional[str] = None
    section: Optional[str] = None


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from kaggle_config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "kaggle_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# BM25 IMPLEMENTATION
# ============================================================================

class BM25Index:
    """
    BM25 (Best Matching 25) index for keyword-based retrieval.
    
    BM25 is a ranking function used for information retrieval that considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across all documents
    - Document length normalization: Adjusts for longer/shorter documents
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        
        # Index data
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        
        # Inverted index: term -> {doc_idx: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        
        # IDF cache
        self.idf_cache: Dict[str, float] = {}
        
        self._indexed = False
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric.
        """
        import re
        # Lowercase and extract words
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return tokens
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs (same order as documents)
        """
        self.documents = documents
        self.doc_ids = doc_ids
        self.doc_count = len(documents)
        
        # Calculate document lengths and build inverted index
        self.doc_lengths = []
        self.inverted_index = {}
        
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            term_freqs = Counter(tokens)
            
            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = freq
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Pre-compute IDF for all terms
        self._compute_idf()
        
        self._indexed = True
        print(f"  BM25 index built: {self.doc_count} documents, {len(self.inverted_index)} unique terms")
    
    def _compute_idf(self):
        """Compute IDF for all terms in the index."""
        self.idf_cache = {}
        for term, doc_freqs in self.inverted_index.items():
            # Number of documents containing the term
            n_docs_with_term = len(doc_freqs)
            # IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)
            idf = math.log((self.doc_count - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)
            self.idf_cache[term] = idf
    
    def get_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        Calculate BM25 scores for all documents given a query.
        
        Args:
            query: Search query text
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not self._indexed:
            return []
        
        query_tokens = self.tokenize(query)
        
        # Calculate BM25 score for each document
        scores = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self.idf_cache.get(term, 0)
            
            for doc_idx, term_freq in self.inverted_index[term].items():
                # BM25 formula
                doc_length = self.doc_lengths[doc_idx]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score = idf * (numerator / denominator)
                
                if doc_idx not in scores:
                    scores[doc_idx] = 0
                scores[doc_idx] += score
        
        # Convert to (doc_id, score) and sort
        results = [(self.doc_ids[doc_idx], score) for doc_idx, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def search(self, query: str, n_results: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        all_scores = self.get_scores(query)
        return all_scores[:n_results]
    
    def save(self, path: Path):
        """Save index to disk."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'doc_count': self.doc_count,
            'inverted_index': self.inverted_index,
            'idf_cache': self.idf_cache,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> 'BM25Index':
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls(k1=data['k1'], b=data['b'])
        index.documents = data['documents']
        index.doc_ids = data['doc_ids']
        index.doc_lengths = data['doc_lengths']
        index.avg_doc_length = data['avg_doc_length']
        index.doc_count = data['doc_count']
        index.inverted_index = data['inverted_index']
        index.idf_cache = data['idf_cache']
        index._indexed = True
        
        return index


# ============================================================================
# HYBRID SEARCH WITH SCORE FUSION
# ============================================================================

class HybridSearcher:
    """
    Hybrid search combining semantic (vector) and keyword (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
    """
    
    def __init__(
        self,
        vector_store: 'KaggleVectorStore',
        embedder: 'KaggleEmbedder',
        bm25_index: Optional[BM25Index] = None,
        semantic_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: Vector database for semantic search
            embedder: Embedding model
            bm25_index: BM25 index for keyword search (will build if None)
            semantic_weight: Weight for semantic vs keyword (0.5 = equal)
            rrf_k: RRF constant (higher = less aggressive rank boosting)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.rrf_k = rrf_k
    
    def build_bm25_index(self):
        """Build BM25 index from vector store documents."""
        print("  Building BM25 index from vector store...")
        
        # Get all documents from vector store
        all_ids = self.vector_store.get_all_ids()
        documents = []
        doc_ids = []
        
        for doc_id in all_ids:
            passage = self.vector_store.get_passage_by_id(doc_id)
            if passage:
                documents.append(passage['text'])
                doc_ids.append(doc_id)
        
        self.bm25_index = BM25Index(k1=1.5, b=0.75)
        self.bm25_index.build_index(documents, doc_ids)
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank)) for each result list
        
        This method is robust and doesn't require score normalization.
        """
        rrf_scores = {}
        
        # Add semantic results (weighted)
        for rank, (doc_id, _) in enumerate(semantic_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += self.semantic_weight * (1.0 / (self.rrf_k + rank + 1))
        
        # Add keyword results (weighted)
        for rank, (doc_id, _) in enumerate(keyword_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += self.keyword_weight * (1.0 / (self.rrf_k + rank + 1))
        
        # Sort by combined RRF score
        results = [(doc_id, score) for doc_id, score in rrf_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def weighted_score_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using weighted score fusion with min-max normalization.
        
        Normalizes scores to [0, 1] range and combines with weights.
        """
        combined_scores = {}
        
        # Normalize and add semantic scores
        if semantic_results:
            # For semantic (distance-based), lower is better - convert to similarity
            distances = [score for _, score in semantic_results]
            max_dist = max(distances) if distances else 1
            min_dist = min(distances) if distances else 0
            dist_range = max_dist - min_dist if max_dist != min_dist else 1
            
            for doc_id, distance in semantic_results:
                # Convert distance to similarity (1 - normalized_distance)
                normalized_sim = 1 - ((distance - min_dist) / dist_range)
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {'semantic': 0, 'keyword': 0}
                combined_scores[doc_id]['semantic'] = normalized_sim
        
        # Normalize and add keyword scores
        if keyword_results:
            scores = [score for _, score in keyword_results]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            score_range = max_score - min_score if max_score != min_score else 1
            
            for doc_id, score in keyword_results:
                normalized_score = (score - min_score) / score_range
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {'semantic': 0, 'keyword': 0}
                combined_scores[doc_id]['keyword'] = normalized_score
        
        # Combine with weights
        results = []
        for doc_id, scores in combined_scores.items():
            combined = (
                self.semantic_weight * scores['semantic'] +
                self.keyword_weight * scores['keyword']
            )
            results.append((doc_id, combined))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword results.
        
        Args:
            query: Search query
            n_results: Number of results to return
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted"
            
        Returns:
            List of passage dictionaries with combined scores
        """
        # Ensure BM25 index exists
        if self.bm25_index is None or not self.bm25_index._indexed:
            self.build_bm25_index()
        
        # Get more candidates than needed for better fusion
        n_candidates = n_results * 3
        
        # Semantic search
        query_embedding = self.embedder.embed_single(query)
        semantic_results = self.vector_store.search(query_embedding, n_results=n_candidates)
        semantic_tuples = [(r['id'], r.get('distance', 0)) for r in semantic_results]
        
        # Keyword search
        keyword_tuples = self.bm25_index.search(query, n_results=n_candidates)
        
        # Fuse results
        if fusion_method == "rrf":
            fused = self.reciprocal_rank_fusion(semantic_tuples, keyword_tuples)
        else:
            fused = self.weighted_score_fusion(semantic_tuples, keyword_tuples)
        
        # Get full passage data for top results
        results = []
        for doc_id, score in fused[:n_results]:
            passage = self.vector_store.get_passage_by_id(doc_id)
            if passage:
                passage['hybrid_score'] = score
                # Check if it appeared in both result sets
                in_semantic = any(doc_id == r[0] for r in semantic_tuples)
                in_keyword = any(doc_id == r[0] for r in keyword_tuples)
                passage['match_type'] = 'both' if (in_semantic and in_keyword) else ('semantic' if in_semantic else 'keyword')
                results.append(passage)
        
        return results


class KaggleVectorStore:
    """Vector store wrapper for Kaggle environment."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize vector store, trying multiple input paths."""
        self.config = load_config(config_path)
        self.vector_db_config = self.config['vector_db']
        self.storage_config = self.config['storage']
        
        # Try to find existing vector DB
        self.persist_dir = None
        self._collection = None
        self._client = None
        
        # Check input paths first, then working directory
        input_paths = self.vector_db_config.get('input_paths', [])
        input_paths.append(self.storage_config['vector_db_dir'])
        
        for path_str in input_paths:
            path = Path(path_str)
            if path.exists() and (path / "chroma.sqlite3").exists():
                self.persist_dir = path
                print(f"Found vector DB at: {path}")
                break
        
        if self.persist_dir is None:
            # Use working directory as fallback
            self.persist_dir = Path(self.storage_config['vector_db_dir'])
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            print(f"Using vector DB directory: {self.persist_dir}")
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB."""
        import chromadb
        from chromadb.config import Settings
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = self.vector_db_config['collection_name']
        try:
            self._collection = self._client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name} ({self._collection.count()} passages)")
        except:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"description": "Mahabharat passages"}
            )
            print(f"Created new collection: {collection_name}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar passages."""
        where = None
        if filter_dict:
            where = {}
            for key, value in filter_dict.items():
                if key not in ["characters", "themes"]:
                    where[key] = value
        
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                }
                if 'characters' in result['metadata']:
                    try:
                        result['metadata']['characters'] = json.loads(result['metadata']['characters'])
                    except:
                        pass
                if 'themes' in result['metadata']:
                    try:
                        result['metadata']['themes'] = json.loads(result['metadata']['themes'])
                    except:
                        pass
                formatted_results.append(result)
        
        return formatted_results
    
    def get_passage_by_id(self, passage_id: str) -> Optional[Dict]:
        """Get a specific passage by ID."""
        results = self._collection.get(ids=[passage_id])
        if results['ids']:
            return {
                'id': results['ids'][0],
                'text': results['documents'][0],
                'metadata': results['metadatas'][0] if 'metadatas' in results else {}
            }
        return None
    
    def get_all_ids(self) -> List[str]:
        """Get all passage IDs."""
        results = self._collection.get()
        return results['ids'] if 'ids' in results else []
    
    def count(self) -> int:
        """Get total number of passages."""
        return self._collection.count()
    
    def add_passages(self, passages: List['Passage']):
        """Add passages to the vector store."""
        if not passages:
            return
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for passage in passages:
            ids.append(passage.chunk_id)
            documents.append(passage.text)
            embeddings.append(passage.embedding.tolist() if isinstance(passage.embedding, np.ndarray) else passage.embedding)
            metadata = {
                'characters': json.dumps(passage.characters) if passage.characters else '[]',
                'themes': json.dumps(passage.themes) if passage.themes else '[]',
            }
            if passage.chapter:
                metadata['chapter'] = passage.chapter
            if passage.section:
                metadata['section'] = passage.section
            metadatas.append(metadata)
        
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )


class KaggleEmbedder:
    """Embedder for Kaggle environment."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize embedder."""
        self.config = load_config(config_path)
        self.embeddings_config = self.config['embeddings']
        self.provider = self.embeddings_config['provider']
        self.model_name = self.embeddings_config['model']
        self.dimension = self.embeddings_config['dimension']
        
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        if self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = self._model.encode(texts, show_progress_bar=len(texts) > 10)
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class KaggleRetriever:
    """
    Retriever adapted for Kaggle with hybrid search support.
    
    Supports:
    - Pure semantic search (original behavior)
    - Pure keyword search (BM25)
    - Hybrid search (semantic + BM25 with score fusion)
    """
    
    def __init__(
        self,
        vector_store: Optional[KaggleVectorStore] = None,
        embedder: Optional[KaggleEmbedder] = None,
        config_path: Optional[Path] = None,
        use_hybrid: Optional[bool] = None,
        semantic_weight: Optional[float] = None
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            config_path: Path to config file
            use_hybrid: Whether to use hybrid search (semantic + BM25), defaults to config
            semantic_weight: Weight for semantic search (0-1), defaults to config
        """
        self.config = load_config(config_path)
        self.vector_store = vector_store or KaggleVectorStore(config_path)
        self.embedder = embedder or KaggleEmbedder(config_path)
        self.diversity_threshold = self.config['retrieval']['diversity_threshold']
        self.max_retries = self.config['retrieval']['max_retries']
        
        # Hybrid search configuration (from config or override)
        retrieval_config = self.config.get('retrieval', {})
        self.use_hybrid = use_hybrid if use_hybrid is not None else retrieval_config.get('use_hybrid', True)
        self.semantic_weight = semantic_weight if semantic_weight is not None else retrieval_config.get('semantic_weight', 0.5)
        self.fusion_method = retrieval_config.get('fusion_method', 'rrf')
        self.rrf_k = retrieval_config.get('rrf_k', 60)
        
        self._hybrid_searcher: Optional[HybridSearcher] = None
        self._bm25_index: Optional[BM25Index] = None
        
        # Initialize hybrid searcher if enabled
        if self.use_hybrid:
            self._init_hybrid_search()
    
    def _init_hybrid_search(self):
        """Initialize hybrid search components."""
        # Try to load existing BM25 index
        bm25_path = Path(self.config['storage']['vector_db_dir']) / "bm25_index.pkl"
        
        if bm25_path.exists():
            try:
                print("Loading existing BM25 index...")
                self._bm25_index = BM25Index.load(bm25_path)
                print(f"  Loaded BM25 index: {self._bm25_index.doc_count} documents")
            except Exception as e:
                print(f"  Failed to load BM25 index: {e}")
                self._bm25_index = None
        
        self._hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            embedder=self.embedder,
            bm25_index=self._bm25_index,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k
        )
        
        # Build BM25 index if not loaded
        if self._bm25_index is None and self.vector_store.count() > 0:
            self._hybrid_searcher.build_bm25_index()
            self._bm25_index = self._hybrid_searcher.bm25_index
            # Save for future use
            try:
                bm25_path.parent.mkdir(parents=True, exist_ok=True)
                self._bm25_index.save(bm25_path)
                print(f"  Saved BM25 index to: {bm25_path}")
            except Exception as e:
                print(f"  Failed to save BM25 index: {e}")
    
    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim
    
    def retrieve_hybrid(
        self,
        query: str,
        n_results: int = 10,
        fusion_method: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve passages using hybrid search (semantic + BM25).
        
        Args:
            query: Search query
            n_results: Number of results
            fusion_method: "rrf" or "weighted" (defaults to config)
            
        Returns:
            List of passage dictionaries with hybrid scores
        """
        if not self._hybrid_searcher:
            self._init_hybrid_search()
        
        method = fusion_method or self.fusion_method
        
        results = self._hybrid_searcher.search(
            query=query,
            n_results=n_results,
            fusion_method=method
        )
        
        return results
    
    def retrieve_diverse(
        self,
        query: Optional[str] = None,
        n_results: int = 1,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve diverse passages using hybrid search when query is provided.
        
        Falls back to random selection when no query is given.
        """
        exclude_ids = exclude_ids or []
        
        if query:
            # Use hybrid search for query-based retrieval
            if self.use_hybrid:
                print(f"  Using hybrid search (semantic: {self.semantic_weight:.0%}, BM25: {1-self.semantic_weight:.0%})")
                candidates = self.retrieve_hybrid(query, n_results=n_results * 10)
                # Log match types for debugging
                match_types = {}
                for c in candidates:
                    mt = c.get('match_type', 'unknown')
                    match_types[mt] = match_types.get(mt, 0) + 1
                print(f"  Match types: {match_types}")
            else:
                # Pure semantic search
                query_embedding = self.embedder.embed_single(query)
                candidates = self.vector_store.search(query_embedding, n_results=n_results * 10)
        else:
            # Random selection (no query)
            all_ids = self.vector_store.get_all_ids()
            available_ids = [id for id in all_ids if id not in exclude_ids]
            
            if not available_ids:
                raise ValueError("No available passages to retrieve")
            
            candidates = []
            np.random.shuffle(available_ids)
            for passage_id in available_ids[:min(100, len(available_ids))]:
                passage = self.vector_store.get_passage_by_id(passage_id)
                if passage:
                    passage['embedding'] = self.embedder.embed_single(passage['text'])
                    candidates.append(passage)
        
        # Filter excluded IDs
        candidates = [c for c in candidates if c['id'] not in exclude_ids]
        
        if not candidates:
            return []
        
        # Apply diversity selection (MMR-like)
        selected = []
        selected_embeddings = []
        
        # Select first candidate (best match)
        if query and self.use_hybrid:
            # Already sorted by hybrid score
            first = candidates[0]
        elif query:
            candidates.sort(key=lambda x: x.get('distance', float('inf')))
            first = candidates[0]
        else:
            first = candidates[0]
        
        selected.append(first)
        if 'embedding' not in first:
            first['embedding'] = self.embedder.embed_single(first['text'])
        selected_embeddings.append(first['embedding'])
        
        remaining = candidates[1:]
        
        # Select remaining candidates with diversity
        for _ in range(min(n_results - 1, len(remaining))):
            best_candidate = None
            best_min_distance = -1
            
            for candidate in remaining:
                if 'embedding' not in candidate:
                    candidate['embedding'] = self.embedder.embed_single(candidate['text'])
                
                min_distance = min(
                    self.cosine_distance(candidate['embedding'], sel_emb)
                    for sel_emb in selected_embeddings
                )
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate and best_min_distance >= self.diversity_threshold:
                selected.append(best_candidate)
                selected_embeddings.append(best_candidate['embedding'])
                remaining.remove(best_candidate)
            else:
                if remaining:
                    selected.append(remaining[0])
                    if 'embedding' not in remaining[0]:
                        remaining[0]['embedding'] = self.embedder.embed_single(remaining[0]['text'])
                    selected_embeddings.append(remaining[0]['embedding'])
                    remaining.pop(0)
        
        # Clean up embeddings from results
        for result in selected:
            if 'embedding' in result:
                del result['embedding']
        
        return selected
    
    def retrieve_context(
        self,
        seed_passage: Dict,
        n_context: int = 3,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Retrieve contextual passages related to a seed passage.
        
        Uses hybrid search to find passages that are both semantically
        similar AND share keywords with the seed passage.
        
        Args:
            seed_passage: The seed passage to find context for
            n_context: Number of context passages to retrieve
            use_hybrid: Whether to use hybrid search for context
            
        Returns:
            List of context passages
        """
        seed_text = seed_passage['text']
        seed_id = seed_passage['id']
        
        if use_hybrid and self.use_hybrid:
            # Use seed text as query for hybrid search
            context_passages = self.retrieve_hybrid(
                query=seed_text,
                n_results=n_context + 5  # Get extra to filter seed
            )
        else:
            # Pure semantic search
            seed_embedding = self.embedder.embed_single(seed_text)
            context_passages = self.vector_store.search(
                seed_embedding,
                n_results=n_context + 1
            )
        
        # Filter out the seed passage itself
        context_passages = [p for p in context_passages if p['id'] != seed_id]
        
        return context_passages[:n_context]


class KaggleStoryGenerator:
    """Story generator adapted for Kaggle."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize story generator."""
        self.config = load_config(config_path)
        self.llm_config = self.config['llm']
        self.story_config = self.config['story']
        
        self.provider = self.llm_config['provider']
        self.model = self.llm_config['model']
        self.temperature = self.llm_config['temperature']
        self.max_tokens = self.llm_config['max_tokens']
        self.api_key_env = self.llm_config['api_key_env']
        self.num_scenes = self.story_config['num_scenes']
        
        self._client = None
        self._initialize_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found! Please set environment variable: {self.api_key_env}\n"
                f"In Kaggle, add it as a secret: Settings → Secrets → Add Secret"
            )
        return api_key
    
    def _initialize_client(self):
        """Initialize LLM client."""
        api_key = self._get_api_key()
        
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_story(
        self,
        seed_passage: str,
        context_passages: List[str]
    ) -> Dict:
        """Generate story sequence from passages."""
        prompt = self._build_prompt(seed_passage, context_passages)
        
        print(f"Generating {self.num_scenes} scenes using {self.provider}...")
        
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        except ImportError:
            safety_settings = None
        
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        response = self._client.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if not getattr(response, 'candidates', None) or len(response.candidates) == 0:
            raise ValueError("LLM returned no candidates - content may have been blocked")
        
        try:
            if hasattr(response, "text"):
                story_text = response.text
            else:
                parts = []
                if getattr(response, "candidates", None):
                    for cand in response.candidates:
                        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                            for p in cand.content.parts:
                                if hasattr(p, "text"):
                                    parts.append(p.text)
                story_text = "\n".join(parts)
        except ValueError as e:
            if "blocked" in str(e).lower():
                raise ValueError("Story content was blocked by safety filter")
            raise
        
        # Parse JSON response
        import re
        story_text = re.sub(r'```(?:json)?\s*', '', story_text)
        story_text = re.sub(r'```\s*$', '', story_text)
        story_text = story_text.strip()
        
        try:
            story_data = json.loads(story_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse preview: {story_text[:500]}")
        
        return story_data
    
    def _build_prompt(self, seed_passage: str, context_passages: List[str]) -> str:
        """Build prompt for story generation."""
        context_text = "\n\n".join([f"Context {i+1}: {p}" for i, p in enumerate(context_passages)])
        
        # Use the same prompt structure as the original StoryGenerator
        prompt = f"""You are creating {self.num_scenes} VIDEO SCENE PROMPTS for a Mahabharat story.

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {self.num_scenes} scenes with FULL character descriptions in EVERY scene
2. Use ACTUAL MAHABHARATA CHARACTERS from the source material (e.g., Arjuna, Krishna, Duryodhana, Bhishma, Karna, Draupadi, Yudhishthira, etc.)
3. Base scenes DIRECTLY on the source material provided - do not create generic characters
4. Each scene must be a specific moment from the Mahabharata story in the source material

SOURCE MATERIAL (MAHABHARATA TEXT):
{seed_passage}

ADDITIONAL CONTEXT:
{context_text}

IMPORTANT: Extract the actual Mahabharata characters, events, and story from the source material above. Use their real names and create scenes that match the specific story being told.

=== CHARACTER DESCRIPTION RULES (MANDATORY) ===

1. FULL DESCRIPTION IN EVERY SCENE - Never use "the same person" or "the same man/woman"
   - Write out the COMPLETE character description in each scene
   - AI video generators cannot reference previous scenes

2. SKIN TONE - Only use these two options:
   - "brown skin" 
   - "fair skin"
   - NEVER use: "dark complexion", "dark skin", "dusky"

3. CLOTHING - Must include in EVERY scene:
   - Specific colors (e.g., "deep red", "golden yellow", "royal blue")
   - Material type (e.g., "silk", "cotton", "leather armor")
   - Style details (e.g., "with gold embroidery", "with silver border")

4. GENDER & AGE - Always specify:
   - "a man in his 40s" or "a woman in her 20s"
   - Never use vague terms like "figure", "person", "warrior"

5. ACTION - Clearly describe what the character is DOING:
   - "raises his right arm", "walks forward slowly", "kneels on the ground"
   - Be specific about body movements

=== OUTPUT FORMAT ===

Return valid JSON with exactly {self.num_scenes} scenes:
{{
  "story_sequence": [
    {{
      "scene_number": 1,
      "title": "Scene Title",
      "narrative": "Narrative description",
      "characters": ["Character1", "Character2"],
      "clip_prompt": {{
        "visual_description": "Full visual description with complete character details",
        "camera_angles": "Camera angles and movements",
        "key_objects": "Important objects",
        "ambient_audio": "Background sounds"
      }}
    }}
  ],
  "metadata": {{
    "characters": ["All", "Characters"],
    "themes": ["Theme1", "Theme2"]
  }}
}}

Generate the story now:"""
        
        return prompt


class KaggleNarrationGenerator:
    """Narration generator (placeholder for audio generation later)."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize narration generator."""
        self.config = load_config(config_path)
        # Placeholder - will be implemented later for audio generation
        pass
    
    def generate_narration(self, script_output: Dict) -> Dict:
        """Generate narration JSON (placeholder)."""
        # For now, return a simple structure
        # Audio generation will be added later
        return {
            "story_title": "Mahabharat Story",
            "hook": "Have you ever wondered about the epic tales of the Mahabharata?",
            "story": " ".join([scene.get('narrative', '') for scene in script_output.get('story_sequence', [])]),
            "cta": "What do you think about this story? Share your thoughts below!",
            "voiceover": "Placeholder narration - audio generation coming soon"
        }

