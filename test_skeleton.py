"""测试骨架可视化功能"""

import cv2
import torch
import numpy as np
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from bvh_export import BVHExporter
from skeleton_visualizer import create_skeleton_video, SkeletonVisualizer

def setup_cfg():
    """设置DensePose配置"""
    cfg = get_cfg()
    add_densepose_config(cfg)
    config_path = Path(__file__).resolve().parent / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def test_skeleton_visualization():
    """测试骨架可视化"""
    print("=== 测试骨架可视化功能 ===\n")
    
    # 初始化
    sample_dir = Path(__file__).resolve().parent / "sample_videos"
    video_files = list(sample_dir.glob("*.mp4"))
    
    if not video_files:
        print("错误: 找不到示例视频")
        return
    
    input_video = str(video_files[0])
    print(f"输入视频: {Path(input_video).name}")
    
    print("初始化DensePose模型...")
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 处理前50帧
    max_frames = 50
    print(f"\n处理前{max_frames}帧...")
    
    densepose_results = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        with torch.no_grad():
            outputs = predictor(frame)
        
        if "instances" in outputs and len(outputs["instances"]) > 0:
            instances = outputs["instances"]
            densepose_results.append(instances)
        else:
            densepose_results.append(None)
        
        frame_count += 1
    
    cap.release()
    
    valid_frames = sum(1 for r in densepose_results if r is not None)
    print(f"✓ 处理完成: {len(densepose_results)}帧, 有效{valid_frames}帧")
    
    # 提取关键点
    print("\n提取骨架关键点...")
    exporter = BVHExporter(fps=fps)
    keypoints_sequence = []
    
    for idx, result in enumerate(densepose_results):
        if result is not None:
            keypoints = exporter.extract_keypoints_from_densepose(result)
            keypoints_sequence.append(keypoints)
            
            if idx == 0 and keypoints is not None:
                print(f"  第0帧关键点示例:")
                print(f"    Hips (0):  {keypoints[0]}")
                print(f"    Neck (3):  {keypoints[3]}")
                print(f"    Head (4):  {keypoints[4]}")
        else:
            keypoints_sequence.append(None)
    
    valid_keypoints = sum(1 for kp in keypoints_sequence if kp is not None)
    print(f"✓ 关键点提取完成: {valid_keypoints}/{len(keypoints_sequence)} 帧")
    
    # 生成骨架可视化视频
    print("\n生成骨架可视化视频...")
    output_path = str(Path(__file__).resolve().parent / "skeleton_test.mp4")
    
    success = create_skeleton_video(
        input_video,
        output_path,
        keypoints_sequence,
        show_labels=False,
        show_legend=True,
        alpha=0.7
    )
    
    if success:
        output_file = Path(output_path)
        print(f"\n✓ 骨架视频已生成: {output_file.name}")
        print(f"  文件大小: {output_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"  位置: {output_file}")
        print("\n你可以使用视频播放器打开此文件查看效果。")
        print("视频中会显示：")
        print("  - 彩色骨骼线条（青色=脊柱，绿色=左臂，红色=右臂，蓝绿色=左腿，紫色=右腿）")
        print("  - 白色关节点")
        print("  - 底部图例说明")
    else:
        print("\n✗ 骨架视频生成失败")
    
    # 测试单帧绘制
    print("\n测试单帧骨架绘制...")
    cap = cv2.VideoCapture(input_video)
    ret, test_frame = cap.read()
    cap.release()
    
    if ret and len(keypoints_sequence) > 0 and keypoints_sequence[0] is not None:
        visualizer = SkeletonVisualizer(line_thickness=4, joint_radius=6)
        skeleton_frame = visualizer.draw_skeleton_with_labels(
            test_frame,
            keypoints_sequence[0],
            show_joint_names=True,
            alpha=0.8
        )
        
        output_img_path = str(Path(__file__).resolve().parent / "skeleton_frame.jpg")
        cv2.imwrite(output_img_path, skeleton_frame)
        print(f"✓ 单帧示例已保存: skeleton_frame.jpg")
        print(f"  （包含关节名称标签）")

if __name__ == "__main__":
    test_skeleton_visualization()
