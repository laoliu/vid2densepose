"""骨架可视化模块 - 在视频上绘制3D骨骼"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class SkeletonVisualizer:
    """在视频帧上绘制骨架结构"""
    
    # 骨架连接关系（基于BVH的19关节结构）
    SKELETON_CONNECTIONS = [
        # 脊柱链
        (0, 1),   # Hips -> Spine
        (1, 2),   # Spine -> Spine1
        (2, 3),   # Spine1 -> Neck
        (3, 4),   # Neck -> Head
        
        # 左臂
        (2, 5),   # Spine1 -> LeftShoulder
        (5, 6),   # LeftShoulder -> LeftArm
        (6, 7),   # LeftArm -> LeftForeArm
        (7, 8),   # LeftForeArm -> LeftHand
        
        # 右臂
        (2, 9),   # Spine1 -> RightShoulder
        (9, 10),  # RightShoulder -> RightArm
        (10, 11), # RightArm -> RightForeArm
        (11, 12), # RightForeArm -> RightHand
        
        # 左腿
        (0, 13),  # Hips -> LeftUpLeg
        (13, 14), # LeftUpLeg -> LeftLeg
        (14, 15), # LeftLeg -> LeftFoot
        
        # 右腿
        (0, 16),  # Hips -> RightUpLeg
        (16, 17), # RightUpLeg -> RightLeg
        (17, 18), # RightLeg -> RightFoot
    ]
    
    # 关节名称
    JOINT_NAMES = [
        "Hips", "Spine", "Spine1", "Neck", "Head",
        "L_Shoulder", "L_Arm", "L_ForeArm", "L_Hand",
        "R_Shoulder", "R_Arm", "R_ForeArm", "R_Hand",
        "L_UpLeg", "L_Leg", "L_Foot",
        "R_UpLeg", "R_Leg", "R_Foot"
    ]
    
    # 颜色方案（BGR格式）
    COLORS = {
        'spine': (255, 200, 0),      # 青色 - 脊柱
        'left_arm': (0, 255, 0),     # 绿色 - 左臂
        'right_arm': (0, 0, 255),    # 红色 - 右臂
        'left_leg': (255, 100, 0),   # 蓝绿色 - 左腿
        'right_leg': (255, 0, 100),  # 紫色 - 右腿
        'joint': (255, 255, 255),    # 白色 - 关节点
    }
    
    def __init__(self, line_thickness: int = 3, joint_radius: int = 5):
        """
        Args:
            line_thickness: 骨骼线条粗细
            joint_radius: 关节点半径
        """
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
    
    def get_bone_color(self, bone_idx: int) -> Tuple[int, int, int]:
        """根据骨骼索引返回颜色"""
        start_joint, end_joint = self.SKELETON_CONNECTIONS[bone_idx]
        
        # 脊柱链 (0-4)
        if start_joint in [0, 1, 2, 3] and end_joint in [1, 2, 3, 4]:
            return self.COLORS['spine']
        
        # 左臂 (5-8)
        elif start_joint in [2, 5, 6, 7] and end_joint in [5, 6, 7, 8]:
            return self.COLORS['left_arm']
        
        # 右臂 (9-12)
        elif start_joint in [2, 9, 10, 11] and end_joint in [9, 10, 11, 12]:
            return self.COLORS['right_arm']
        
        # 左腿 (13-15)
        elif start_joint in [0, 13, 14] and end_joint in [13, 14, 15]:
            return self.COLORS['left_leg']
        
        # 右腿 (16-18)
        elif start_joint in [0, 16, 17] and end_joint in [16, 17, 18]:
            return self.COLORS['right_leg']
        
        return (200, 200, 200)  # 默认灰色
    
    def draw_skeleton(
        self, 
        frame: np.ndarray, 
        keypoints: np.ndarray,
        alpha: float = 0.8
    ) -> np.ndarray:
        """
        在帧上绘制骨架
        
        Args:
            frame: 输入帧 (H, W, 3)
            keypoints: 关键点坐标 (19, 3) - [x, y, z]
            alpha: 透明度 (0-1)
        
        Returns:
            绘制了骨架的帧
        """
        if keypoints is None or len(keypoints) != 19:
            return frame
        
        # 创建叠加层
        overlay = frame.copy()
        
        # 绘制骨骼连接线
        for bone_idx, (start_idx, end_idx) in enumerate(self.SKELETON_CONNECTIONS):
            pt1 = keypoints[start_idx][:2].astype(int)  # [x, y]
            pt2 = keypoints[end_idx][:2].astype(int)
            
            # 确保点在图像范围内
            if (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0] and
                0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                
                color = self.get_bone_color(bone_idx)
                cv2.line(overlay, tuple(pt1), tuple(pt2), color, self.line_thickness)
        
        # 绘制关节点
        for joint_idx, joint_pos in enumerate(keypoints):
            pt = joint_pos[:2].astype(int)
            
            if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                # 外圈（深色）
                cv2.circle(overlay, tuple(pt), self.joint_radius + 1, (0, 0, 0), -1)
                # 内圈（白色）
                cv2.circle(overlay, tuple(pt), self.joint_radius, self.COLORS['joint'], -1)
        
        # 混合原图和叠加层
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def draw_skeleton_with_labels(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        show_joint_names: bool = False,
        alpha: float = 0.8
    ) -> np.ndarray:
        """
        绘制带标签的骨架
        
        Args:
            frame: 输入帧
            keypoints: 关键点坐标
            show_joint_names: 是否显示关节名称
            alpha: 透明度
        
        Returns:
            绘制了骨架和标签的帧
        """
        result = self.draw_skeleton(frame, keypoints, alpha)
        
        if show_joint_names and keypoints is not None:
            # 绘制关节名称
            for joint_idx, joint_pos in enumerate(keypoints):
                pt = joint_pos[:2].astype(int)
                
                if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                    name = self.JOINT_NAMES[joint_idx]
                    
                    # 文字背景
                    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(
                        result,
                        (pt[0] - 2, pt[1] - text_size[1] - 4),
                        (pt[0] + text_size[0] + 2, pt[1] - 2),
                        (0, 0, 0),
                        -1
                    )
                    
                    # 文字
                    cv2.putText(
                        result,
                        name,
                        (pt[0], pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
        
        return result
    
    def add_legend(self, frame: np.ndarray) -> np.ndarray:
        """添加图例说明"""
        legend_height = 150
        legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
        
        y_start = 30
        line_spacing = 25
        
        legend_items = [
            ("Spine", self.COLORS['spine']),
            ("Left Arm", self.COLORS['left_arm']),
            ("Right Arm", self.COLORS['right_arm']),
            ("Left Leg", self.COLORS['left_leg']),
            ("Right Leg", self.COLORS['right_leg']),
        ]
        
        for idx, (label, color) in enumerate(legend_items):
            y = y_start + idx * line_spacing
            
            # 绘制颜色线
            cv2.line(legend, (20, y), (60, y), color, self.line_thickness)
            
            # 绘制文字
            cv2.putText(
                legend,
                label,
                (70, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        # 拼接到原图下方
        result = np.vstack([frame, legend])
        return result


def create_skeleton_video(
    input_video_path: str,
    output_video_path: str,
    keypoints_sequence: List[Optional[np.ndarray]],
    show_labels: bool = False,
    show_legend: bool = False,
    alpha: float = 0.8
) -> bool:
    """
    创建带骨架可视化的视频
    
    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        keypoints_sequence: 每帧的关键点序列
        show_labels: 是否显示关节名称
        show_legend: 是否显示图例
        alpha: 骨架透明度
    
    Returns:
        是否成功
    """
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果显示图例，增加输出高度
    output_height = height + 150 if show_legend else height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, output_height))
    
    visualizer = SkeletonVisualizer()
    
    frame_idx = 0
    while frame_idx < len(keypoints_sequence):
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = keypoints_sequence[frame_idx]
        
        if keypoints is not None:
            if show_labels:
                frame = visualizer.draw_skeleton_with_labels(frame, keypoints, True, alpha)
            else:
                frame = visualizer.draw_skeleton(frame, keypoints, alpha)
        
        if show_legend:
            frame = visualizer.add_legend(frame)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"✓ 骨架视频已生成: {output_video_path}")
    return True
