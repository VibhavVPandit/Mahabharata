# Wan 2.1 Video Generation on Kaggle

This directory contains everything needed to generate videos from Mahabharata stories using Wan 2.1 on Kaggle.

## Overview

The solution generates videos directly from Mahabharata stories:

1. **Create Vector Database** from PDF (first time only)
2. **Retrieve passages** from vector database
3. **Generate story scripts** (~5 scenes) and narration using Gemini API
4. **Generate videos** from scripts using Wan 2.1

## Files

- `wan21_video_generation.ipynb` - Main Kaggle notebook
- `wan21_generator.py` - Core video generation module
- `story_pipeline.py` - Story generation from vector DB
- `kaggle_config.yaml` - Configuration file
- `requirements_kaggle.txt` - Python dependencies

## Setup Instructions

### 1. Push to GitHub

Push the `kaggle/` directory to your GitHub repository:

```bash
git add kaggle/
git commit -m "Add Kaggle video generation setup"
git push origin main
```

### 2. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Click **"Add Data"** → **"GitHub"**
4. Enter your repository URL: `https://github.com/VibhavVPandit/Mahabharata`
5. Click **"Add"**

### 3. Configure Notebook Settings

In the sidebar, set:

- **Accelerator**: **GPU T4 x2** (required for dual GPU support)
- **Language**: **Python 3**
- **Internet**: **On** (required for model downloads)
- **Persistence**: **Files Only** (optional, keeps downloaded models)

### 4. Upload PDF

1. Click **"Add Data"** → **"Upload"**
2. Upload `Mahabharata.pdf` to `/kaggle/input/`

### 5. Set API Key

1. Go to **Settings** → **Secrets** → **Add Secret**
2. **Name**: `GEMINI_API_KEY`
3. **Value**: Your Gemini API key
4. Click **"Add"**

## Usage

1. Open the notebook `wan21_video_generation.ipynb`
2. In the **Configuration** cell, optionally set:
   ```python
   QUERY = "Arjuna"  # Or None for random passage
   PDF_PATH = "/kaggle/input/Mahabharata.pdf"  # Update if different
   ```
3. Run all cells in order:
   - **Cell 1**: Setup and installation
   - **Cell 2**: Configuration
   - **Cell 3**: Create vector database from PDF (first time only, skips if already exists)
   - **Cell 4**: Generate story scripts and videos
   - **Cell 5**: Zip output for download
4. The notebook will:
   - Extract text from PDF and create vector database (if needed)
   - Retrieve passages from vector database
   - Generate ~5 scenes and narration using Gemini API
   - Save scripts: `video_prompts_TIMESTAMP.json` and `narration_TIMESTAMP.json`
   - Generate videos from scenes
5. Download videos and scripts from `/kaggle/working/output/`

## Technical Specifications

### Hardware Requirements

- **Accelerator**: NVIDIA Tesla T4 x2 (Dual GPU)
- **VRAM**: ~8.2 GB to 11 GB per GPU
- **System RAM**: 30 GB
- **Disk Space**: Minimum 20 GB free

### Model Configuration

- **Model**: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- **Resolution**: 832x480
- **Frames**: 49, 81, or 121 (follows `(n - 1) % 4 == 0` rule)
- **FPS**: 16
- **Inference Steps**: 50
- **Sample Shift**: 4.0
- **Scheduler**: UniPCMultistepScheduler

### Memory Management

- Uses `enable_model_cpu_offload()` for dual GPU support
- Hugging Face cache in `/kaggle/tmp/` (not counted against working directory limit)
- Transformer: `torch.bfloat16`
- VAE: `torch.float32` (prevents flickering artifacts)

## Output

All outputs are saved to `/kaggle/working/output/`:

- **Videos**: `scene_01.mp4`, `scene_02.mp4`, etc.
- **Scripts**: `video_prompts_TIMESTAMP.json` (video prompts for each scene)
- **Narration**: `narration_TIMESTAMP.json` (story title, hook, story, CTA, voiceover)

You can download individual files or use the zip cell to create `videos_output.zip`.

## Troubleshooting

### Vector Database Not Found

**Error**: "Vector database is empty"

**Solution**: 
1. Ensure `Mahabharata.pdf` is uploaded to `/kaggle/input/`
2. Run Cell 3 (Create Vector Database from PDF) first
3. Wait for the vector database creation to complete before running Cell 4

### Out of Memory

**Error**: CUDA out of memory

**Solution**:
1. Ensure you're using **GPU T4 x2** (not single GPU)
2. Check that `enable_model_cpu_offload()` is being used
3. Reduce `num_inference_steps` in `kaggle_config.yaml` (try 30 instead of 50)

### Model Download Slow

**Issue**: Initial model download takes 5-8 minutes

**Solution**: This is normal. The model (~2-3 GB) downloads on first run. Subsequent runs use cached version.

### API Key Not Found

**Error**: "API key not found"

**Solution**:
1. Go to Settings → Secrets
2. Ensure `GEMINI_API_KEY` is set
3. Restart the notebook session

## Known Limitations

1. **Timeouts**: Generation of long clips (120+ frames) may approach Kaggle's session timeout limits
2. **Cold Start**: Initial model download takes 5-8 minutes
3. **Audio Narration**: Currently generates JSON only. Audio file generation (TTS) will be added later

