"""Story generation pipeline adapted for Kaggle environment."""
import os
import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import adapted components
import sys
# Add parent directory to path to import from src if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from kaggle_config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "kaggle_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    """Retriever adapted for Kaggle."""
    
    def __init__(
        self,
        vector_store: Optional[KaggleVectorStore] = None,
        embedder: Optional[KaggleEmbedder] = None,
        config_path: Optional[Path] = None
    ):
        """Initialize retriever."""
        self.config = load_config(config_path)
        self.vector_store = vector_store or KaggleVectorStore(config_path)
        self.embedder = embedder or KaggleEmbedder(config_path)
        self.diversity_threshold = self.config['retrieval']['diversity_threshold']
        self.max_retries = self.config['retrieval']['max_retries']
    
    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim
    
    def retrieve_diverse(
        self,
        query: Optional[str] = None,
        n_results: int = 1,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve diverse passages."""
        exclude_ids = exclude_ids or []
        
        if query:
            query_embedding = self.embedder.embed_single(query)
            candidates = self.vector_store.search(query_embedding, n_results=n_results * 10)
        else:
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
        
        candidates = [c for c in candidates if c['id'] not in exclude_ids]
        
        if not candidates:
            return []
        
        selected = []
        selected_embeddings = []
        
        if query:
            candidates.sort(key=lambda x: x.get('distance', float('inf')))
            first = candidates[0]
        else:
            first = candidates[0]
        
        selected.append(first)
        if 'embedding' not in first:
            first['embedding'] = self.embedder.embed_single(first['text'])
        selected_embeddings.append(first['embedding'])
        
        remaining = candidates[1:]
        
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
        
        for result in selected:
            if 'embedding' in result:
                del result['embedding']
        
        return selected
    
    def retrieve_context(
        self,
        seed_passage: Dict,
        n_context: int = 3
    ) -> List[Dict]:
        """Retrieve contextual passages."""
        seed_text = seed_passage['text']
        seed_id = seed_passage['id']
        
        seed_embedding = self.embedder.embed_single(seed_text)
        context_passages = self.vector_store.search(
            seed_embedding,
            n_results=n_context + 1
        )
        
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

