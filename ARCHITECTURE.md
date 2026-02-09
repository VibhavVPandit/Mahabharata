# Architecture & Technical Documentation

This document provides detailed technical information about how the Mahabharata Story-to-Video Generation Pipeline works under the hood.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hybrid Search System](#hybrid-search-system)
3. [Vector Database](#vector-database)
4. [Story Generation Pipeline](#story-generation-pipeline)
5. [Video Generation Pipeline](#video-generation-pipeline)
6. [Component Details](#component-details)

---

## System Architecture

### High-Level Flow

```
┌─────────────────┐
│  PDF Document   │
│ (Mahabharata)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction│
│  & Chunking     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Database│
│  (ChromaDB)     │
│  + BM25 Index   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid Search  │
│  (Query/Random) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Passage        │
│  Retrieval      │
│  (Seed+Context) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Story          │
│  Generation     │
│  (Gemini API)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Video Script   │
│  (5 Scenes)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Video          │
│  Generation     │
│  (Wan 2.1)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Videos  │
│  + Scripts      │
└─────────────────┘
```

### Component Interaction

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ KaggleVectorStore│◄────►│ KaggleEmbedder  │◄────►│ KaggleRetriever  │
│  (ChromaDB)      │      │ (Embeddings)    │      │ (Hybrid Search)  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                        │                           │
         │                        │                           │
         └────────────────────────┴───────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ KaggleStoryGenerator    │
                    │   (Gemini API)          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Wan21KaggleGenerator  │
                    │   (Video Synthesis)    │
                    └────────────────────────┘
```

---

## Hybrid Search System

### Overview

The hybrid search system combines semantic (vector) and keyword (BM25) retrieval methods to achieve better search results than either method alone.

### Architecture

```
Query Input
    │
    ├──► Semantic Search ──┐
    │     (Vector Embedding)│
    │                       │
    └──► Keyword Search ────┤
          (BM25 Index)      │
                            ▼
                    ┌───────────────┐
                    │ Score Fusion  │
                    │ (RRF/Weighted)│
                    └───────┬───────┘
                            ▼
                    Ranked Results
```

### Semantic Search

**Process:**
1. Query text is converted to embedding vector using `all-MiniLM-L6-v2` model
2. Embedding is compared against all passage embeddings in ChromaDB
3. Cosine similarity is calculated: `similarity = (A · B) / (||A|| × ||B||)`
4. Results are ranked by similarity (distance)

**Implementation:**
```python
# Query embedding
query_embedding = embedder.embed_single(query)

# Vector search
results = vector_store.search(
    query_embedding,
    n_results=n_candidates
)
```

**Advantages:**
- Captures semantic meaning (synonyms, related concepts)
- Works well with paraphrased queries
- Understands context and relationships

**Limitations:**
- May miss exact keyword matches
- Can be sensitive to embedding quality

### Keyword Search (BM25)

**BM25 Algorithm:**
BM25 (Best Matching 25) is a probabilistic ranking function that scores documents based on:
- **Term Frequency (TF)**: How often a term appears in a document
- **Inverse Document Frequency (IDF)**: How rare a term is across all documents
- **Document Length Normalization**: Adjusts for longer/shorter documents

**Formula:**
```
BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d|/avgdl))

Where:
- q = query
- d = document
- f(qi, d) = term frequency of qi in d
- |d| = document length
- avgdl = average document length
- k1 = term frequency saturation (default: 1.5)
- b = length normalization (default: 0.75)
```

**Process:**
1. Tokenize query and documents (lowercase, extract words)
2. Build inverted index: `term → {doc_id: frequency}`
3. Calculate IDF for each term: `log((N - n + 0.5) / (n + 0.5) + 1)`
4. Score each document using BM25 formula
5. Rank by score

**Implementation:**
```python
class BM25Index:
    def build_index(self, documents, doc_ids):
        # Build inverted index
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_idx] = freq
        
        # Calculate IDF
        self._compute_idf()
    
    def search(self, query, n_results=10):
        query_tokens = self.tokenize(query)
        scores = {}
        for term in query_tokens:
            idf = self.idf_cache[term]
            for doc_idx, term_freq in self.inverted_index[term].items():
                # BM25 calculation
                score = idf * (term_freq * (k1 + 1)) / ...
                scores[doc_idx] += score
        return sorted results
```

**Advantages:**
- Excellent for exact keyword matches
- Fast and efficient
- Works well with specific names, places, terms

**Limitations:**
- Misses semantic relationships
- No understanding of synonyms
- Requires exact term match

### Score Fusion Methods

#### Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

Where:
- k = RRF constant (default: 60)
- rank_i(d) = rank of document d in result set i
- weight_i = weight for method i
```

**Process:**
1. Get ranked results from both semantic and keyword search
2. For each document, calculate RRF score from both rankings
3. Combine scores: `combined = semantic_weight × RRF_semantic + keyword_weight × RRF_keyword`
4. Re-rank by combined score

**Advantages:**
- No score normalization needed
- Robust to different score scales
- Simple and effective

**Implementation:**
```python
def reciprocal_rank_fusion(semantic_results, keyword_results):
    rrf_scores = {}
    
    # Add semantic results
    for rank, (doc_id, _) in enumerate(semantic_results):
        rrf_scores[doc_id] = semantic_weight / (rrf_k + rank + 1)
    
    # Add keyword results
    for rank, (doc_id, _) in enumerate(keyword_results):
        rrf_scores[doc_id] += keyword_weight / (rrf_k + rank + 1)
    
    return sorted by score
```

#### Weighted Score Fusion

**Process:**
1. Normalize scores from both methods to [0, 1] range
2. Apply weights: `combined = w_semantic × score_semantic + w_keyword × score_keyword`
3. Rank by combined score

**Normalization:**
- Semantic (distance-based): Convert to similarity: `1 - normalized_distance`
- Keyword (score-based): Min-max normalization: `(score - min) / (max - min)`

**Advantages:**
- More control over weighting
- Can emphasize one method over another

**Disadvantages:**
- Requires score normalization
- Sensitive to score distributions

### Diversity Filtering

After hybrid search, results are filtered for diversity using MMR-like algorithm:

1. Select best match (highest score)
2. For remaining slots:
   - Calculate minimum distance to already-selected passages
   - Select passage with maximum minimum distance
   - Only if distance > diversity_threshold (default: 0.3)

This ensures retrieved passages are not too similar to each other.

---

## Vector Database

### ChromaDB Integration

**Storage Structure:**
```
Collection: "mahabharat_passages"
├── IDs: Unique passage identifiers
├── Documents: Text content
├── Embeddings: 384-dimensional vectors
└── Metadata:
    ├── characters: JSON array
    ├── themes: JSON array
    ├── chapter: String (optional)
    └── section: String (optional)
```

### Embedding Generation

**Model:** `all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Provider:** sentence-transformers
- **Properties:**
  - Fast inference
  - Good semantic understanding
  - Balanced performance/speed

**Process:**
1. Text is cleaned and normalized
2. Passed to SentenceTransformer model
3. Generates 384-dimensional embedding vector
4. Stored in ChromaDB with metadata

### Chunking Strategy

**Parameters:**
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 50 characters

**Rationale:**
- Preserves context across chunk boundaries
- Balances granularity with context preservation
- Optimal for embedding models

**Implementation:**
```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context
    return chunks
```

### Metadata Extraction

**Characters:**
- Extracted from text using pattern matching
- Stored as JSON array in metadata
- Used for filtering and retrieval

**Themes:**
- Identified from context
- Tagged for thematic search
- Stored as JSON array

---

## Story Generation Pipeline

### Process Flow

```
Seed Passage + Context Passages
         │
         ▼
┌────────────────────┐
│ Prompt Engineering │
│  (Structured)      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Gemini API Call   │
│  (Gemini 2.5 Flash)│
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  JSON Response     │
│  (5 Scenes)        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Parse & Validate  │
└────────────────────┘
```

### Prompt Structure

The prompt is carefully engineered to ensure:
1. **Character Consistency**: Uses actual Mahabharata characters from source
2. **Full Descriptions**: Every scene has complete character descriptions (video generators can't reference previous scenes)
3. **Visual Details**: Specific skin tones, clothing, actions
4. **Structured Output**: JSON format with exact schema

**Key Prompt Sections:**
- Source material (seed passage)
- Additional context (3 related passages)
- Character description rules
- Output format specification

### LLM Configuration

**Model:** Google Gemini 2.5 Flash
- **Temperature:** 0.7 (balanced creativity/consistency)
- **Max Tokens:** 8000
- **Safety Settings:** Configured for story content

**Response Format:**
```json
{
  "story_sequence": [
    {
      "scene_number": 1,
      "title": "Scene Title",
      "narrative": "Story narrative",
      "characters": ["Character1", "Character2"],
      "clip_prompt": {
        "visual_description": "Full visual description",
        "camera_angles": "Camera details",
        "key_objects": "Important objects",
        "ambient_audio": "Background sounds"
      }
    }
  ],
  "metadata": {
    "characters": ["All", "Characters"],
    "themes": ["Theme1", "Theme2"]
  }
}
```

### Error Handling

- **Content Blocking:** Handles safety filter blocks
- **JSON Parsing:** Validates and cleans JSON response
- **Retry Logic:** Configurable retry attempts for API failures

---

## Video Generation Pipeline

### Wan 2.1 Model Architecture

**Model:** Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- **Type:** Diffusion-based text-to-video
- **Resolution:** 832x480
- **FPS:** 16
- **Parameters:** 1.3B

### Frame Count Validation

**Constraint:** `(n - 1) % 4 == 0`

This constraint is required by the Wan 2.1 model architecture. Valid frame counts:
- 49 frames (3.06 seconds)
- 81 frames (5.06 seconds)
- 121 frames (7.56 seconds)

**Calculation:**
```python
def calculate_frames(duration_seconds, fps=16):
    target_frames = int(duration_seconds * fps)
    # Find closest valid frame count
    base = (target_frames - 1) // 4
    frames = base * 4 + 1
    
    # Clamp to valid range [49, 121]
    frames = max(49, min(121, frames))
    
    return frames
```

### Memory Management

**Dual GPU Strategy:**
- Uses `enable_model_cpu_offload()` for automatic GPU distribution
- Transformer: `torch.bfloat16` (memory efficiency)
- VAE: `torch.float32` (prevents flickering artifacts)

**Why VAE in float32:**
- bfloat16 can cause flickering/artifacts in video generation
- VAE is smaller, so float32 is acceptable
- Transformer is larger, benefits from bfloat16

### Generation Process

```
Visual Description Prompt
         │
         ▼
┌────────────────────┐
│  Frame Calculation │
│  (49/81/121)       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Diffusion Process │
│  (50 steps)        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Frame Extraction  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Video Encoding    │
│  (MP4, 16fps)      │
└────────────────────┘
```

### Scheduler Configuration

**Scheduler:** UniPCMultistepScheduler
- Optimized for video generation
- Provides smooth temporal transitions
- Configurable inference steps (default: 50)

**Sample Shift:** 4.0
- Controls temporal consistency
- Higher values = more variation
- Lower values = more consistency

### Pipeline Configuration

```python
pipeline = Wan2_1Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)

# Configure scheduler
pipeline.scheduler = UniPCMultistepScheduler.from_config(
    pipeline.scheduler.config
)

# Enable CPU offloading for dual GPU
pipeline.enable_model_cpu_offload()

# Set VAE to float32
pipeline.vae.to(dtype=torch.float32)
```

### Video Export

**Format:** MP4 (H.264)
- **FPS:** 16
- **Resolution:** 832x480
- **Codec:** H.264

**Process:**
1. Generated frames are extracted from pipeline output
2. Frames are encoded to video using `imageio` and `ffmpeg`
3. Saved as `scene_XX.mp4` files

---

## Component Details

### KaggleVectorStore

**Purpose:** ChromaDB wrapper for vector storage

**Key Methods:**
- `search(query_embedding, n_results)`: Semantic search
- `add_passages(passages)`: Add passages with embeddings
- `get_passage_by_id(id)`: Retrieve specific passage
- `get_all_ids()`: Get all passage IDs
- `count()`: Total passage count

**Features:**
- Automatic path detection (checks multiple input paths)
- Metadata support (characters, themes, chapters)
- Persistent storage

### KaggleEmbedder

**Purpose:** Generate embeddings using sentence-transformers

**Key Methods:**
- `embed(texts)`: Batch embedding generation
- `embed_single(text)`: Single text embedding

**Model:** `all-MiniLM-L6-v2`
- Lightweight and fast
- 384 dimensions
- Good semantic understanding

### KaggleRetriever

**Purpose:** Hybrid search with diversity filtering

**Key Methods:**
- `retrieve_hybrid(query, n_results)`: Hybrid search
- `retrieve_diverse(query, n_results)`: Diverse passage selection
- `retrieve_context(seed_passage, n_context)`: Context retrieval

**Features:**
- Configurable hybrid search (semantic/keyword weights)
- MMR-like diversity filtering
- Automatic BM25 index building

### KaggleStoryGenerator

**Purpose:** Generate story scripts using Gemini API

**Key Methods:**
- `generate_story(seed_passage, context_passages)`: Main generation method
- `_build_prompt(seed, context)`: Prompt construction
- `_initialize_client()`: API client setup

**Features:**
- Structured prompt engineering
- JSON response parsing
- Error handling for API failures

### Wan21KaggleGenerator

**Purpose:** Video generation using Wan 2.1 model

**Key Methods:**
- `generate_scene_video(prompt, scene_number, duration)`: Single scene generation
- `generate_from_json(json_path)`: Batch generation from JSON
- `generate_from_scenes(scenes)`: Generate from scene objects
- `calculate_frames(duration)`: Frame count calculation
- `validate_frame_count(n)`: Frame validation

**Features:**
- Automatic frame count calculation
- Dual GPU support
- Quality optimizations (VAE float32)
- Batch processing support

---

## Performance Considerations

### Memory Usage

- **Transformer:** ~4-5 GB per GPU (bfloat16)
- **VAE:** ~1-2 GB per GPU (float32)
- **Total:** ~8-11 GB per GPU
- **System RAM:** ~30 GB recommended

### Speed

- **Embedding Generation:** ~100-200 passages/second
- **Vector Search:** <100ms for 10k passages
- **Story Generation:** ~5-10 seconds (API dependent)
- **Video Generation:** ~30-60 seconds per scene (50 steps)

### Optimization Strategies

1. **Caching:** Vector DB and BM25 index are persistent
2. **Batch Processing:** Embeddings generated in batches
3. **GPU Offloading:** Automatic distribution across GPUs
4. **Model Quantization:** bfloat16 for transformer

---

## Configuration

All configuration is managed through `kaggle_config.yaml`:

- **Storage Paths:** Cache, output, vector DB directories
- **Model Settings:** Model IDs, resolutions, inference steps
- **Search Parameters:** Hybrid weights, fusion method, diversity threshold
- **Generation Settings:** Number of scenes, temperature, max tokens

See `kaggle_config.yaml` for all available options with inline comments.

