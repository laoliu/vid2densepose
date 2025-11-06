import argparse
import os

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def main(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    
    # Use local config file
    config_file = os.path.join(os.path.dirname(__file__), "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.merge_from_file(config_file)
    
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer with H.264 codec for better compatibility
    # Try different codecs in order of preference
    output_temp = output_video_path.replace('.mp4', '_temp.mp4')
    fourcc_options = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    for fourcc_code in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {fourcc_code}")
            break
    
    if not out or not out.isOpened():
        raise RuntimeError("Failed to initialize video writer with any codec")

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            outputs = predictor(frame)["instances"]

        results = DensePoseResultExtractor()(outputs)

        # MagicAnimate uses the Viridis colormap for their training data
        cmap = cv2.COLORMAP_VIRIDIS
        # Visualizer outputs black for background, but we want the 0 value of
        # the colormap, so we initialize the array with that value
        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
        out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
        out.write(out_frame)

    # Release resources
    cap.release()
    out.release()
    
    print(f"\nVideo processing complete!")
    print(f"Output saved to: {output_video_path}")
    
    # Check if we should try to convert for better compatibility
    if output_temp != output_video_path and os.path.exists(output_temp):
        # Try to use ffmpeg for better compatibility
        import subprocess
        try:
            print("\nConverting to H.264 for better compatibility...")
            result = subprocess.run([
                'ffmpeg', '-i', output_temp, '-c:v', 'libx264', 
                '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-y', output_video_path
            ], check=True, capture_output=True, text=True)
            os.remove(output_temp)
            print("✓ Video converted successfully!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # If ffmpeg fails or is not available, just rename the temp file
            print("\nNote: ffmpeg not available for conversion.")
            print("If the video doesn't play, you can convert it using:")
            print(f"  python convert_video.py {output_temp}")
            if os.path.exists(output_temp):
                os.rename(output_temp, output_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    args = parser.parse_args()

    main(args.input_video_path, args.output_video_path)
