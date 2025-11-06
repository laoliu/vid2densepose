#!/usr/bin/env python3
"""
Convert video to a more compatible format using ffmpeg
"""
import subprocess
import sys
import os

def convert_video(input_file, output_file=None):
    """Convert video to H.264 MP4 format for maximum compatibility"""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return False
    
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_converted.mp4"
    
    print(f"Converting {input_file} to {output_file}...")
    
    try:
        # Use ffmpeg to convert to H.264 with yuv420p pixel format for compatibility
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',      # Use H.264 codec
            '-preset', 'medium',     # Encoding speed/quality tradeoff
            '-crf', '23',            # Quality (lower = better, 23 is default)
            '-pix_fmt', 'yuv420p',   # Pixel format for compatibility
            '-y',                    # Overwrite output file
            output_file
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully converted to: {output_file}")
        return True
        
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install it:")
        print("  sudo apt-get install ffmpeg")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Conversion failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_video.py <input_video> [output_video]")
        print("\nExample:")
        print("  python convert_video.py sample_videos/output_video.mp4")
        print("  python convert_video.py sample_videos/output_video.mp4 sample_videos/final.mp4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_video(input_file, output_file)
    sys.exit(0 if success else 1)
