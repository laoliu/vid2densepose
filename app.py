import os
from pathlib import Path
from typing import Union

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import gradio_client.utils as gr_utils
from detectron2.config import get_cfg
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
import tempfile
from bvh_export import create_bvh_from_densepose_results, BVHExporter
from skeleton_visualizer import create_skeleton_video


_original_json_schema_to_python_type = gr_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "bool"
    return _original_json_schema_to_python_type(schema, defs)


gr_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

def _resolve_input_path(input_video: Union[str, dict, None]) -> Path:
    """Normalize Gradio video input into a local file path."""
    if input_video is None:
        raise gr.Error("请先上传或录制一段视频。")

    if isinstance(input_video, str):
        candidate = Path(input_video)
    elif isinstance(input_video, dict):
        candidate = Path(input_video.get("path") or input_video.get("name", ""))
    else:
        raise gr.Error("暂不支持该类型的视频输入。")

    if not candidate.exists():
        raise gr.Error(f"未找到视频文件: {candidate}")

    return candidate


# Function to process video
def process_video(input_video_path, export_bvh: bool = False, show_skeleton: bool = False):
    progress = gr.Progress(track_tqdm=True)

    src_path = _resolve_input_path(input_video_path)

    # Temporary path for output video
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    skeleton_video_path = None
    bvh_output_path = None
    
    if export_bvh:
        bvh_output_path = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False).name
    
    if show_skeleton:
        skeleton_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    config_path = Path(__file__).resolve().parent / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
    cfg.merge_from_file(config_path.as_posix())
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    progress(0.05, desc="加载 DensePose 模型")
    predictor = DefaultPredictor(cfg)

    # Open the input video
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise gr.Error("无法读取视频，请确认文件格式或重新录制后再试。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Store DensePose results for BVH export / skeleton visualization
    densepose_results = [] if (export_bvh or show_skeleton) else None
    keypoints_sequence = [] if show_skeleton else None

    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            outputs = predictor(frame)
        
        # Extract instances
        if "instances" not in outputs or len(outputs["instances"]) == 0:
            # No detection, write original frame
            out.write(frame)
            if export_bvh:
                densepose_results.append(None)
        else:
            instances = outputs["instances"]
            results = DensePoseResultExtractor()(instances)
            
            # Store instances for BVH export (not the extractor result)
            if export_bvh or show_skeleton:
                densepose_results.append(instances)
            
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)        
            out.write(out_frame)

        frame_idx += 1
        if total_frames > 0:
            progress(0.05 + 0.9 * (frame_idx / total_frames), desc=f"处理帧 {frame_idx}/{total_frames}")
        else:
            progress(0.05, desc=f"处理帧 {frame_idx}")

    # Release resources
    cap.release()
    out.release()

    progress(0.95, desc="导出视频结果")
    
    # Extract keypoints for skeleton visualization
    if show_skeleton and densepose_results:
        progress(0.96, desc="提取骨架关键点")
        exporter = BVHExporter(fps=fps)
        for result in densepose_results:
            if result is not None:
                keypoints = exporter.extract_keypoints_from_densepose(result)
                keypoints_sequence.append(keypoints)
            else:
                keypoints_sequence.append(None)
    
    # Export BVH if requested
    if export_bvh and densepose_results:
        progress(0.97, desc="生成 BVH 文件")
        try:
            success = create_bvh_from_densepose_results(
                densepose_results, 
                bvh_output_path,
                fps=fps,
                scale=0.1
            )
            if success:
                progress(0.99, desc="BVH 导出完成")
        except Exception as e:
            print(f"BVH 导出失败: {e}")
            bvh_output_path = None
    
    # Create skeleton visualization video if requested
    if show_skeleton and keypoints_sequence:
        progress(0.98, desc="生成骨架可视化视频")
        try:
            success = create_skeleton_video(
                str(src_path),
                skeleton_video_path,
                keypoints_sequence,
                show_labels=False,
                show_legend=True,
                alpha=0.7
            )
            if not success:
                skeleton_video_path = None
        except Exception as e:
            print(f"骨架视频生成失败: {e}")
            skeleton_video_path = None

    progress(1.0, desc="完成")

    # Return processed videos and BVH file
    return output_video_path, skeleton_video_path, bvh_output_path

# Gradio interface
root_dir = Path(__file__).resolve().parent
sample_dir = root_dir / "sample_videos"
example_videos = [[str(p.resolve())] for p in sample_dir.glob("*.mp4")]

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(
            label="Input Video",
            sources=["upload", "webcam"],
            include_audio=False,
        ),
        gr.Checkbox(
            label="导出 BVH 文件",
            value=False,
            info="生成骨骼动画 BVH 文件（可用于 Blender、Maya 等 3D 软件）"
        ),
        gr.Checkbox(
            label="显示骨架可视化",
            value=True,
            info="在视频上叠加 3D 骨架模型（彩色骨骼线条）"
        )
    ],
    outputs=[
        gr.Video(label="DensePose 输出（体表分割）"),
        gr.Video(label="骨架可视化输出（3D 骨骼）", visible=True),
        gr.File(label="BVH 动作文件", visible=True)
    ],
    title="Video 2 DensePose + 3D Skeleton + BVH",
    description="""
    上传视频或使用摄像头录制，自动生成：
    
    1. **DensePose 体表分割** - 彩色人体表面映射
    2. **3D 骨架可视化** - 在视频上叠加骨骼模型（勾选「显示骨架可视化」）
    3. **BVH 动作文件** - 可导入 Blender/Unity/Maya（勾选「导出 BVH 文件」）
    
    示例视频如下：
    """,
    examples=example_videos,
    allow_flagging="never",
    concurrency_limit=1,
)

# Run the app with a public share link for environments without localhost access
iface.launch(share=True, show_api=False)
