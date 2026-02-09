"""Wan 2.1 Video Generator for Kaggle."""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch
from diffusers import Wan2_1Pipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from kaggle_config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "kaggle_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Wan21KaggleGenerator:
    """Generate videos using Wan 2.1 model on Kaggle."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Wan 2.1 generator with configuration."""
        self.config = load_config(config_path)
        self.wan21_config = self.config['wan21']
        self.storage_config = self.config['storage']
        self.validation_config = self.config['validation']
        
        # Setup paths
        self.output_dir = Path(self.storage_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Hugging Face cache
        cache_dir = Path(self.storage_config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
        
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Wan 2.1 pipeline with proper settings."""
        print("Initializing Wan 2.1 pipeline...")
        print(f"Model: {self.wan21_config['model_id']}")
        print(f"Resolution: {self.wan21_config['resolution']}")
        
        # Load pipeline
        self.pipeline = Wan2_1Pipeline.from_pretrained(
            self.wan21_config['model_id'],
            torch_dtype=torch.bfloat16,
            cache_dir=self.storage_config['cache_dir']
        )
        
        # Set scheduler
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # Enable CPU offloading for dual GPU support
        self.pipeline.enable_model_cpu_offload()
        
        # Set VAE to float32 to avoid flickering artifacts
        if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
            self.pipeline.vae.to(dtype=torch.float32)
        
        print("Pipeline initialized successfully!")
    
    def validate_frame_count(self, n: int) -> bool:
        """
        Validate frame count follows the rule: (n - 1) % 4 == 0.
        
        Args:
            n: Number of frames
            
        Returns:
            True if valid, False otherwise
        """
        return (n - 1) % self.validation_config['frame_rule_modulo'] == 0
    
    def calculate_frames(self, duration_seconds: float) -> int:
        """
        Calculate valid frame count from duration.
        
        Formula: frames = ((duration_seconds * fps) // 4) * 4 + 1
        This ensures (n - 1) % 4 == 0
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Valid frame count
        """
        fps = self.wan21_config['output_fps']
        target_frames = int(duration_seconds * fps)
        
        # Find the closest valid frame count
        base = (target_frames - 1) // 4
        frames = base * 4 + 1
        
        # Ensure within bounds
        min_frames = self.validation_config['min_frames']
        max_frames = self.validation_config['max_frames']
        
        if frames < min_frames:
            frames = min_frames
        elif frames > max_frames:
            frames = max_frames
        
        # Validate
        if not self.validate_frame_count(frames):
            # Fallback: round up to next valid count
            frames = ((frames - 1) // 4 + 1) * 4 + 1
        
        return frames
    
    def generate_scene_video(
        self,
        prompt: str,
        scene_number: int,
        duration_seconds: float = 5.0,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate video for a single scene.
        
        Args:
            prompt: Video generation prompt
            scene_number: Scene number (for filename)
            duration_seconds: Duration in seconds
            output_dir: Output directory (defaults to configured output_dir)
            
        Returns:
            Path to generated video file
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate frame count
        num_frames = self.calculate_frames(duration_seconds)
        
        if not self.validate_frame_count(num_frames):
            raise ValueError(f"Invalid frame count: {num_frames}. Must satisfy (n - 1) % 4 == 0")
        
        print(f"\nGenerating Scene {scene_number}...")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {duration_seconds}s at {self.wan21_config['output_fps']} fps")
        
        # Get resolution
        width, height = self.wan21_config['resolution']
        
        # Generate video
        output = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=self.wan21_config['num_inference_steps'],
            sample_shift=self.wan21_config['sample_shift'],
            generator=torch.Generator().manual_seed(42)  # Reproducibility
        )
        
        # Extract frames (pipeline returns dict or object with frames attribute)
        if isinstance(output, dict):
            video_frames = output.get('frames', output.get('images', output))
        elif hasattr(output, 'frames'):
            video_frames = output.frames
        else:
            video_frames = output
        
        # Export to video
        output_path = output_dir / f"scene_{scene_number:02d}.mp4"
        export_to_video(
            video_frames,
            str(output_path),
            fps=self.wan21_config['output_fps']
        )
        
        print(f"  ✓ Saved: {output_path}")
        return output_path
    
    def generate_from_json(
        self,
        json_path: Union[str, Path],
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Generate videos from JSON file (Mode 1).
        
        Args:
            json_path: Path to video_prompts JSON file
            output_dir: Output directory (defaults to configured output_dir)
            
        Returns:
            List of paths to generated video files
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        print(f"\nLoading video prompts from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        
        if not isinstance(scenes, list):
            raise ValueError("JSON must contain a list of scenes")
        
        print(f"Found {len(scenes)} scenes to generate")
        
        video_paths = []
        for scene_data in scenes:
            scene_number = scene_data.get('scene_number', len(video_paths) + 1)
            prompt = scene_data.get('video_prompt', '')
            duration = scene_data.get('duration_seconds', 5.0)
            
            if not prompt:
                print(f"  ⚠ Skipping scene {scene_number}: No video_prompt found")
                continue
            
            try:
                video_path = self.generate_scene_video(
                    prompt=prompt,
                    scene_number=scene_number,
                    duration_seconds=duration,
                    output_dir=output_dir
                )
                video_paths.append(video_path)
            except Exception as e:
                print(f"  ✗ Error generating scene {scene_number}: {e}")
                continue
        
        print(f"\n✓ Generated {len(video_paths)} videos successfully")
        return video_paths
    
    def generate_from_scenes(
        self,
        scenes: List[Dict],
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Generate videos from list of Scene objects (Mode 2).
        
        Args:
            scenes: List of scene dictionaries with clip_prompt data
            output_dir: Output directory (defaults to configured output_dir)
            
        Returns:
            List of paths to generated video files
        """
        print(f"\nGenerating videos from {len(scenes)} scenes...")
        
        video_paths = []
        for scene in scenes:
            # Extract scene data
            if isinstance(scene, dict):
                scene_number = scene.get('scene_number', len(video_paths) + 1)
                
                # Handle different scene formats
                if 'clip_prompt' in scene:
                    # ScriptOutput format
                    clip_prompt = scene['clip_prompt']
                    if isinstance(clip_prompt, dict):
                        prompt = clip_prompt.get('visual_description', '')
                    else:
                        prompt = str(clip_prompt)
                elif 'video_prompt' in scene:
                    # Direct video_prompt format
                    prompt = scene['video_prompt']
                else:
                    print(f"  ⚠ Skipping scene {scene_number}: No prompt found")
                    continue
                
                duration = scene.get('duration_seconds', 5.0)
            else:
                # Assume it's a Scene object with attributes
                scene_number = getattr(scene, 'scene_number', len(video_paths) + 1)
                clip_prompt = getattr(scene, 'clip_prompt', None)
                if clip_prompt:
                    prompt = getattr(clip_prompt, 'visual_description', '')
                else:
                    prompt = ''
                duration = 5.0
            
            if not prompt:
                print(f"  ⚠ Skipping scene {scene_number}: No prompt found")
                continue
            
            try:
                video_path = self.generate_scene_video(
                    prompt=prompt,
                    scene_number=scene_number,
                    duration_seconds=duration,
                    output_dir=output_dir
                )
                video_paths.append(video_path)
            except Exception as e:
                print(f"  ✗ Error generating scene {scene_number}: {e}")
                continue
        
        print(f"\n✓ Generated {len(video_paths)} videos successfully")
        return video_paths

