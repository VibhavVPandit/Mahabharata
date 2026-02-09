# Mahabharata Story-to-Video Generation Pipeline

An AI-powered pipeline that transforms Mahabharata text into video narratives using hybrid search, LLM story generation, and state-of-the-art video synthesis.

## 🎯 Features

### **Hybrid Search System**
- **Semantic Search**: Meaning-based retrieval using vector embeddings
- **Keyword Search**: Exact term matching with BM25 algorithm
- **Intelligent Fusion**: Combines both methods for optimal results
- **Diverse Retrieval**: Ensures varied passage selection

### **Intelligent Story Generation**
- **Context-Aware**: Retrieves relevant passages from vector database
- **LLM-Powered**: Uses Google Gemini API to generate structured story sequences
- **Character-Aware**: Extracts and maintains character consistency from source material
- **Scene Generation**: Creates 5-scene video scripts with detailed visual descriptions

### **Video Generation**
- **Wan 2.1 Model**: State-of-the-art text-to-video generation
- **Dual GPU Support**: Optimized for Kaggle's T4 x2 GPU configuration
- **Frame Validation**: Automatic frame count calculation
- **Quality Optimization**: Prevents flickering artifacts

### **Vector Database**
- **ChromaDB Integration**: Persistent vector storage with metadata
- **Semantic Chunking**: Intelligent text segmentation preserving context
- **Metadata Extraction**: Character and theme tagging for enhanced retrieval

## 🔄 Workflow

```
PDF Document → Text Extraction → Vector Database → Hybrid Search → 
Passage Retrieval → Story Generation → Video Script → Video Generation → Output
```

1. **Upload PDF**: Provide your Mahabharata PDF document
2. **Create Vector Database**: System processes and indexes the text (first time only)
3. **Search**: Query for specific characters/events or use random selection
4. **Generate Story**: AI creates a 5-scene story script with visual descriptions
5. **Generate Videos**: Each scene is converted to a video using Wan 2.1 model
6. **Download**: Get your video files and story scripts

## 📁 Project Structure

```
kaggle/
├── story_pipeline.py          # Vector DB, retrieval, story generation
├── wan21_generator.py          # Video generation with Wan 2.1
├── kaggle_config.yaml          # Configuration file
├── requirements_kaggle.txt     # Python dependencies
├── wan21_video_generation.ipynb # Main Kaggle notebook
├── COLAB_SETUP.md             # Google Colab setup guide
└── ARCHITECTURE.md            # Technical documentation
```

## 🚀 Quick Start

### **Prerequisites**
- Kaggle account with GPU access (T4 x2 recommended)
- Gemini API key
- PDF document (Mahabharata or similar)

### **Setup**
1. Upload files to Kaggle notebook
2. Add `GEMINI_API_KEY` to Kaggle Secrets
3. Upload PDF to `/kaggle/input/`
4. Run notebook cells sequentially

### **Configuration**
Edit `kaggle_config.yaml` to customize:
- Story length (number of scenes)
- Search weights (semantic vs keyword)
- Video resolution and frame count
- Model parameters

## 📊 Technical Specifications

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM**: Google Gemini 2.5 Flash
- **Video Model**: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- **Vector DB**: ChromaDB (persistent storage)
- **Resolution**: 832x480 @ 16fps
- **Frame Options**: 49, 81, or 121 frames (3-7 seconds)

## 📝 Output Format

The pipeline generates:
- **Video Files**: `scene_01.mp4`, `scene_02.mp4`, etc.
- **Video Scripts**: `video_prompts_TIMESTAMP.json` with detailed scene descriptions
- **Narration**: `narration_TIMESTAMP.json` (for future TTS integration)

## 🎓 Use Cases

- **Educational Content**: Transform historical texts into visual narratives
- **Storytelling**: Create video stories from literary sources
- **Content Creation**: Generate video content from text documents
- **Research**: Visualize narrative structures from large text corpora

## 📚 Documentation

- **ARCHITECTURE.md**: Detailed technical documentation on backend and video generation
- **COLAB_SETUP.md**: Google Colab setup instructions
- **kaggle_config.yaml**: Configuration options with inline comments

---
