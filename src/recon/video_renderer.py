"""
VideoRenderer — Assembles animated face frames + audio into an MP4 video.

Supports two modes:
  1. generate_video_2d() — Takes pre-rendered 2D frames (from FaceAnimator)
  2. generate_video()   — Legacy 3D mesh rendering (kept for compatibility)
"""

import os
import io
import time
import numpy as np
from PIL import Image

try:
    from moviepy import ImageSequenceClip, AudioFileClip
except ImportError:
    try:
        from moviepy.editor import ImageSequenceClip, AudioFileClip
    except ImportError:
        pass

from tqdm import tqdm


class VideoRenderer:
    def __init__(self, output_dir="assets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("VideoRenderer initialized.")

    def generate_video_2d(self, frames, audio_path, fps=30, filename="speaking_avatar.mp4"):
        """
        Generates an MP4 video from pre-rendered 2D face frames + audio.
        
        frames:     list of numpy arrays (RGB images)
        audio_path: path to the WAV audio file
        fps:        frames per second
        filename:   output filename
        """
        if not frames:
            raise ValueError("No frames provided. Ensure face animation ran correctly.")
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Assembling {len(frames)} frames into video...")
        
        # Create Video Clip
        clip = ImageSequenceClip(frames, fps=fps)
        
        # Add Audio
        audio_clip = AudioFileClip(audio_path)
        
        if audio_clip.duration > clip.duration:
            audio_clip = audio_clip.subclipped(0, clip.duration)
        
        clip = clip.with_audio(audio_clip)
        
        # Write to file
        clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )
        
        clip.close()
        audio_clip.close()
        
        print(f"Video saved to {output_path}")
        return output_path

    def generate_video(self, fitter, audio_path, expressions, fps=30, filename="speaking_avatar.mp4", max_frames=None):
        """
        Legacy: Generates video by modifying 3D mesh parameters per frame.
        Kept for backward compatibility.
        """
        import trimesh
        
        output_path = os.path.join(self.output_dir, filename)
        self.base_camera_transform = None
        
        if max_frames and len(expressions) > max_frames:
            expressions = expressions[:max_frames]
            print(f"Truncated to {max_frames} frames.")
            
        print(f"Rendering {len(expressions)} frames for video...")
        
        frames = []
        
        for val in tqdm(expressions, desc="Rendering Frames"):
            expr_deltas = {
                2: float(val) * 0.8,
                1: float(val) * 0.15,
                0: float(val) * 0.1
            }
            
            try:
                frame_mesh = fitter.modify_params(expr_deltas=expr_deltas, refine=False)
                if frame_mesh is None:
                    print("modify_params returned None. Generate an avatar first!")
                    break
                    
                img = self._render_mesh_to_image(frame_mesh)
                frames.append(np.array(img))
            except Exception as e:
                print(f"Error modifying params or rendering: {e}")
                break
                
        if not frames:
            raise ValueError("No frames were rendered. Ensure 3D avatar is generated first.")
            
        print("Assembling video with MoviePy...")
        
        clip = ImageSequenceClip(frames, fps=fps)
        audio_clip = AudioFileClip(audio_path)
        
        if audio_clip.duration > clip.duration:
            audio_clip = audio_clip.subclipped(0, clip.duration)
            
        clip = clip.with_audio(audio_clip)
        
        clip.write_videofile(
            output_path, 
            fps=fps, 
            codec="libx264", 
            audio_codec="aac", 
            logger=None
        )
        
        clip.close()
        audio_clip.close()
        
        print(f"Video saved to {output_path}")
        return output_path

    def _render_mesh_to_image(self, mesh, resolution=(512, 512)):
        """Legacy 3D mesh rendering (fallback)."""
        import trimesh
        
        try:
            if hasattr(mesh.visual, 'to_color'):
                mesh.visual = mesh.visual.to_color()
        except Exception:
            pass
        
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        
        if self.base_camera_transform is None:
            scene.set_camera()
            self.base_camera_transform = scene.camera_transform.copy()
        
        scene.camera_transform = self.base_camera_transform
        
        try:
            png_bytes = scene.save_image(resolution=resolution, visible=True)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return img
        except Exception:
            return Image.new('RGB', resolution, (255, 255, 255))
