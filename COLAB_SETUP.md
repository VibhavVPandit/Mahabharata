# Google Colab Setup Guide

This guide will help you set up and test the improved pipeline with hybrid search and semantic chunking on Google Colab.

## Prerequisites

1. **Google Colab Notebook** (with GPU enabled)
2. **Google Drive** (for storing files)
3. **Gemini API Key** (for story generation)

---

## Step 1: Upload Files to Colab

### Option A: Upload via Colab UI

1. In your Colab notebook, click the **folder icon** (📁) in the left sidebar
2. Click **"Upload to session storage"**
3. Upload these files from your `kaggle/` directory:
   - `story_pipeline.py`
   - `wan21_generator.py`
   - `kaggle_config.yaml`
   - `requirements_kaggle.txt`
   - `wan21_video_generation.ipynb` (or create new cells)
   - `Mahabharata.pdf` (your source PDF)

### Option B: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive to Colab
!cp -r /content/drive/MyDrive/your_project/kaggle/* /content/
```

---

## Step 2: Create Colab-Compatible Config

The paths in `kaggle_config.yaml` are Kaggle-specific. Create a Colab version:

```python
# Create colab_config.yaml
colab_config = """
# Storage Configuration (Colab-specific paths)
storage:
  cache_dir: "/content/cache/huggingface"
  output_dir: "/content/output"
  vector_db_dir: "/content/vector_db"
  input_dir: "/content/input"

# ... (rest of config stays the same)
"""
```

Or modify the config programmatically:

```python
import yaml
from pathlib import Path

# Load original config
with open('kaggle_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update paths for Colab
config['storage']['cache_dir'] = '/content/cache/huggingface'
config['storage']['output_dir'] = '/content/output'
config['storage']['vector_db_dir'] = '/content/vector_db'
config['storage']['input_dir'] = '/content/input'

# Update vector_db paths
config['vector_db']['persist_directory'] = '/content/vector_db'
config['vector_db']['input_paths'] = [
    '/content/input/vector_db',
    '/content/vector_db'
]

# Save Colab config
with open('colab_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

---

## Step 3: Install Dependencies

```python
# Install all required packages
!pip install -q diffusers>=0.21.0 transformers>=4.30.0 accelerate>=0.20.0
!pip install -q imageio>=2.31.0 imageio-ffmpeg>=0.4.8 sentencepiece
!pip install -q chromadb>=0.4.15 sentence-transformers>=2.2.0
!pip install -q google-generativeai>=0.3.0 pydantic>=2.0.0 pyyaml>=6.0.1 numpy>=1.24.0
!pip install -q pypdf>=3.0.0

print("✓ Dependencies installed")
```

---

## Step 4: Set Up Environment

```python
import os
from pathlib import Path

# Create necessary directories
Path('/content/cache/huggingface').mkdir(parents=True, exist_ok=True)
Path('/content/output').mkdir(parents=True, exist_ok=True)
Path('/content/vector_db').mkdir(parents=True, exist_ok=True)
Path('/content/input').mkdir(parents=True, exist_ok=True)

# Set Hugging Face cache
os.environ['HF_HOME'] = '/content/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/content/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/content/cache/huggingface'

# Set Gemini API key (replace with your key)
os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key-here'

print("✓ Environment configured")
```

---

## Step 5: Upload PDF

```python
# Option 1: Upload via Colab UI
# Use the file upload button in the sidebar

# Option 2: Copy from Drive
!cp /content/drive/MyDrive/Mahabharata.pdf /content/input/

# Option 3: Download from URL (if available)
# !wget -O /content/input/Mahabharata.pdf "your-pdf-url"
```

---

## Step 6: Test the Pipeline

### Cell 1: Setup and Imports

```python
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/content')

# Import modules
from story_pipeline import (
    KaggleVectorStore,
    KaggleEmbedder,
    KaggleRetriever,
    KaggleStoryGenerator,
    KaggleNarrationGenerator
)
from wan21_generator import Wan21KaggleGenerator

# Configuration
config_path = Path('/content/colab_config.yaml')  # or 'kaggle_config.yaml' if you updated paths
PDF_PATH = '/content/input/Mahabharata.pdf'
QUERY = "Arjuna"  # or None for random

print("✓ Imports successful")
```

### Cell 2: Create Vector Database

```python
# This will use the new hybrid chunking and improved cleaning
exec(open('/content/wan21_video_generation.ipynb').read())
# Or copy the relevant cells from the notebook

# Alternatively, run the pipeline code directly:
from story_pipeline import KaggleVectorStore, KaggleEmbedder, Passage
# ... (copy the vector DB creation code from Cell 21 of the notebook)
```

### Cell 3: Test Hybrid Search

```python
# Initialize components
vector_store = KaggleVectorStore(config_path=config_path)
embedder = KaggleEmbedder(config_path=config_path)
retriever = KaggleRetriever(
    vector_store=vector_store,
    embedder=embedder,
    config_path=config_path,
    use_hybrid=True  # Enable hybrid search
)

# Test hybrid search
query = "Arjuna battles Karna"
results = retriever.retrieve_hybrid(query, n_results=5)

print(f"\nFound {len(results)} results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['id']}")
    print(f"   Score: {result.get('hybrid_score', 'N/A'):.4f}")
    print(f"   Match: {result.get('match_type', 'unknown')}")
    print(f"   Text: {result['text'][:150]}...")
```

### Cell 4: Generate Story and Videos

```python
# Retrieve seed passage
seed_passages = retriever.retrieve_diverse(query=QUERY, n_results=1)
seed_passage = seed_passages[0]

# Retrieve context
context_passages = retriever.retrieve_context(seed_passage, n_context=3)

# Generate story
story_generator = KaggleStoryGenerator(config_path=config_path)
story_data = story_generator.generate_story(
    seed_passage=seed_passage['text'],
    context_passages=[p['text'] for p in context_passages]
)

print(f"✓ Generated story with {len(story_data.get('story_sequence', []))} scenes")
```

---

## Step 7: Verify Hybrid Search is Working

```python
# Check if BM25 index was created
bm25_path = Path('/content/vector_db/bm25_index.pkl')
if bm25_path.exists():
    print("✓ BM25 index found - hybrid search is active")
else:
    print("⚠ BM25 index not found - will be created on first search")

# Test both search methods
query = "Krishna"

# Pure semantic
semantic_results = vector_store.search(
    embedder.embed_single(query),
    n_results=5
)

# Hybrid search
hybrid_results = retriever.retrieve_hybrid(query, n_results=5)

print(f"\nSemantic only: {len(semantic_results)} results")
print(f"Hybrid search: {len(hybrid_results)} results")

# Compare top results
print("\nTop semantic result:")
print(semantic_results[0]['text'][:200])

print("\nTop hybrid result:")
print(hybrid_results[0]['text'][:200])
```

---

## Troubleshooting

### Issue: "Module not found"
```python
# Make sure files are in /content directory
!ls /content/*.py

# Add to path
import sys
sys.path.insert(0, '/content')
```

### Issue: "Config file not found"
```python
# Check if config exists
from pathlib import Path
config_path = Path('/content/kaggle_config.yaml')
if not config_path.exists():
    print("Config not found. Update paths manually or create colab_config.yaml")
```

### Issue: "BM25 index build slow"
- This is normal on first run. The index is saved and reused on subsequent runs.

### Issue: "Out of memory"
- Colab free tier has limited RAM. Consider:
  - Using smaller chunk sizes
  - Processing in batches
  - Using Colab Pro for more resources

---

## Quick Test Script

Here's a complete test script you can run:

```python
# Complete test pipeline
import sys
from pathlib import Path

sys.path.insert(0, '/content')

from story_pipeline import KaggleVectorStore, KaggleEmbedder, KaggleRetriever

# Setup
config_path = Path('/content/kaggle_config.yaml')
vector_store = KaggleVectorStore(config_path=config_path)
embedder = KaggleEmbedder(config_path=config_path)
retriever = KaggleRetriever(
    vector_store=vector_store,
    embedder=embedder,
    config_path=config_path,
    use_hybrid=True
)

# Test search
results = retriever.retrieve_hybrid("Arjuna", n_results=3)
print(f"✓ Hybrid search working! Found {len(results)} results")
for r in results:
    print(f"  - {r['id']}: {r.get('match_type', 'unknown')} match")
```

---

## Next Steps

1. ✅ Test hybrid search with different queries
2. ✅ Verify chunking quality (check chunk sizes and metadata)
3. ✅ Generate a complete story with videos
4. ✅ Compare results with/without hybrid search

Good luck! 🚀

