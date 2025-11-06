"""快速测试：只处理前30帧"""

import cv2
import torch
import numpy as np
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
from bvh_export import create_bvh_from_densepose_results

def setup_cfg():
    cfg = get_cfg()
    add_densepose_config(cfg)
    config_path = Path(__file__).resolve().parent / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def quick_test():
    sample_dir = Path(__file__).resolve().parent / "sample_videos"
    video_files = list(sample_dir.glob("*.mp4"))
    if not video_files:
        print("错误: 找不到示例视频")
        return
    
    input_video = str(video_files[0])
    print(f"输入视频: {Path(input_video).name}")
    print("初始化模型...")
    
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频: {width}x{height} @ {fps:.2f} FPS")
    
    output_path = str(Path(__file__).resolve().parent / "quick_test.mp4")
    bvh_path = str(Path(__file__).resolve().parent / "quick_test.bvh")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    densepose_results = []
    max_frames = 30
    
    print(f"\n处理前{max_frames}帧...")
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        with torch.no_grad():
            outputs = predictor(frame)
        
        if "instances" in outputs and len(outputs["instances"]) > 0:
            instances = outputs["instances"]
            results = DensePoseResultExtractor()(instances)
            densepose_results.append(instances)
            
            cmap = cv2.COLORMAP_VIRIDIS
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
            out.write(out_frame)
            
            if frame_idx == 0:
                print(f"  帧0: 检测到 {len(instances)} 个人")
        else:
            out.write(frame)
            densepose_results.append(None)
            if frame_idx == 0:
                print(f"  帧0: 未检测到人")
    
    cap.release()
    out.release()
    
    valid_frames = sum(1 for r in densepose_results if r is not None)
    print(f"✓ 处理完成: {len(densepose_results)}帧, 有效{valid_frames}帧")
    print(f"  输出视频: {output_path}")
    
    print("\n导出BVH...")
    success = create_bvh_from_densepose_results(densepose_results, bvh_path, fps=fps, scale=0.1)
    
    if success:
        bvh_file = Path(bvh_path)
        print(f"✓ BVH导出成功: {bvh_file.name}")
        print(f"  大小: {bvh_file.stat().st_size / 1024:.1f} KB")
        
        with open(bvh_path, 'r') as f:
            lines = f.readlines()
        
        print(f"  行数: {len(lines)}")
        print("\n前10行:")
        for line in lines[:10]:
            print(f"  {line.rstrip()}")
        
        print("\n最后2行:")
        for line in lines[-2:]:
            print(f"  {line.rstrip()}")
    else:
        print("✗ BVH导出失败")

if __name__ == "__main__":
    quick_test()
