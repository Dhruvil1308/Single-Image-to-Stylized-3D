"""
Test script for the Lip Sync V2 pipeline (2D Face Animation).
Tests FaceAnimator + AudioSynchronizer + VideoRenderer end-to-end.
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Testing Lip Sync V2 — 2D Face Animation Pipeline")
    print("=" * 60)
    
    os.makedirs("assets", exist_ok=True)
    
    # Step 1: Load source face image
    from PIL import Image
    from src.preprocess.extractor import FaceExtractor
    
    extractor = FaceExtractor()
    input_path = "temp_input.jpg"
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Please place a face photo there.")
        return
    
    print("\n[1/5] Aligning face...")
    aligned = extractor.align_and_crop(input_path)
    print(f"  Aligned image: {aligned.size}")
    
    # Step 2: Initialize FaceAnimator and detect landmarks
    from src.models.face_animator import FaceAnimator
    
    print("\n[2/5] Detecting face landmarks with MediaPipe...")
    animator = FaceAnimator()
    success = animator.set_source(aligned)
    if not success:
        print("ERROR: Could not detect face landmarks!")
        return
    
    # Step 3: Generate TTS audio
    from src.models.audio_sync import AudioSynchronizer
    
    audio_sync = AudioSynchronizer()
    audio_path = os.path.abspath("assets/speech.wav")
    test_text = "Hello! I am your 3D avatar speaking with realistic lip sync."
    
    print("\n[3/5] Generating speech audio...")
    t0 = time.time()
    audio_sync.generate_speech(test_text, audio_path)
    print(f"  Audio generated in {time.time() - t0:.1f}s")
    
    # Step 4: Extract envelope and generate frames
    print("\n[4/5] Extracting envelope and generating animation frames...")
    t0 = time.time()
    envelope = audio_sync.extract_envelope(audio_path, fps=30, max_mouth_open=2.0)
    print(f"  Envelope: {len(envelope)} frames")
    
    frames = animator.generate_frames(envelope, fps=30)
    print(f"  Generated {len(frames)} frames in {time.time() - t0:.1f}s")
    
    # Save a sample frame for visual inspection
    if frames:
        Image.fromarray(frames[len(frames) // 2]).save("assets/sample_frame.png")
        print("  Sample frame saved to assets/sample_frame.png")
    
    # Step 5: Assemble video
    from src.recon.video_renderer import VideoRenderer
    
    print("\n[5/5] Assembling final video...")
    renderer = VideoRenderer()
    t0 = time.time()
    output_path = renderer.generate_video_2d(
        frames, audio_path, fps=30, filename="speaking_avatar.mp4"
    )
    print(f"  Video assembled in {time.time() - t0:.1f}s")
    
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Video saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
