"""
BVH Export Module for DensePose Results
Converts DensePose detection results to BVH motion capture format
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class BVHSkeleton:
    """BVH骨骼层次结构定义"""
    
    # 标准人体骨骼层次（符合BVH格式）
    HIERARCHY = {
        "Hips": {
            "offset": [0, 0, 0],
            "channels": ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
            "children": {
                "Spine": {
                    "offset": [0, 6, 0],
                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                    "children": {
                        "Spine1": {
                            "offset": [0, 6, 0],
                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                            "children": {
                                "Neck": {
                                    "offset": [0, 6, 0],
                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                    "children": {
                                        "Head": {
                                            "offset": [0, 4, 0],
                                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                            "children": {}
                                        }
                                    }
                                },
                                "LeftShoulder": {
                                    "offset": [4, 4, 0],
                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                    "children": {
                                        "LeftArm": {
                                            "offset": [8, 0, 0],
                                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                            "children": {
                                                "LeftForeArm": {
                                                    "offset": [10, 0, 0],
                                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                                    "children": {
                                                        "LeftHand": {
                                                            "offset": [8, 0, 0],
                                                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                                            "children": {}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "RightShoulder": {
                                    "offset": [-4, 4, 0],
                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                    "children": {
                                        "RightArm": {
                                            "offset": [-8, 0, 0],
                                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                            "children": {
                                                "RightForeArm": {
                                                    "offset": [-10, 0, 0],
                                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                                    "children": {
                                                        "RightHand": {
                                                            "offset": [-8, 0, 0],
                                                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                                            "children": {}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "LeftUpLeg": {
                    "offset": [4, -2, 0],
                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                    "children": {
                        "LeftLeg": {
                            "offset": [0, -16, 0],
                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                            "children": {
                                "LeftFoot": {
                                    "offset": [0, -16, 0],
                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                    "children": {}
                                }
                            }
                        }
                    }
                },
                "RightUpLeg": {
                    "offset": [-4, -2, 0],
                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                    "children": {
                        "RightLeg": {
                            "offset": [0, -16, 0],
                            "channels": ["Zrotation", "Xrotation", "Yrotation"],
                            "children": {
                                "RightFoot": {
                                    "offset": [0, -16, 0],
                                    "channels": ["Zrotation", "Xrotation", "Yrotation"],
                                    "children": {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


class BVHExporter:
    """BVH文件导出器"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.skeleton = BVHSkeleton()
        
    def extract_keypoints_from_densepose(self, densepose_result) -> Optional[np.ndarray]:
        """
        从DensePose结果中提取关键点
        
        Args:
            densepose_result: DensePose检测结果 (Instances对象)
            
        Returns:
            关键点坐标数组 shape: (num_joints, 3) 或 None
        """
        if densepose_result is None:
            return None
            
        try:
            # 检查是否有检测结果
            if not hasattr(densepose_result, 'pred_boxes') or len(densepose_result) == 0:
                return None
            
            # 获取第一个检测到的人体（如果有多个，只处理第一个）
            pred_boxes = densepose_result.pred_boxes.tensor.cpu().numpy()
            if len(pred_boxes) == 0:
                return None
            
            # 获取DensePose预测结果
            pred_densepose = densepose_result.pred_densepose
            
            # 获取边界框
            box = pred_boxes[0]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            # DensePose身体部位索引映射 (SMPL 24 parts)
            # 参考: https://github.com/facebookresearch/DensePose/blob/main/configs/densepose_rcnn_R_50_FPN_s1x.yaml
            # 身体部位: 0=背景, 1-24=不同身体部位
            body_part_groups = {
                'head': [23, 24],           # 头部
                'torso': [1, 2],            # 躯干
                'left_arm': [15, 16, 17, 18, 19, 20],  # 左臂
                'right_arm': [3, 4, 5, 6, 7, 8],       # 右臂
                'left_leg': [21, 22],       # 左腿
                'right_leg': [9, 10],       # 右腿
            }
            
            # 从fine_segm提取身体部位中心点
            fine_segm = pred_densepose.fine_segm[0].cpu().numpy()  # [25, 112, 112]
            
            # 将分割图坐标映射回原图
            seg_h, seg_w = 112, 112
            scale_x = width / seg_w
            scale_y = height / seg_h
            
            def get_body_part_center(part_indices):
                """计算身体部位的中心点"""
                mask = np.zeros((seg_h, seg_w), dtype=bool)
                for idx in part_indices:
                    if idx < fine_segm.shape[0]:
                        part_mask = fine_segm[idx] > 0.5
                        mask |= part_mask
                
                if mask.sum() == 0:
                    return None
                
                # 计算质心
                y_coords, x_coords = np.where(mask)
                cy = y_coords.mean()
                cx = x_coords.mean()
                
                # 映射到原图坐标
                real_x = x1 + cx * scale_x
                real_y = y1 + cy * scale_y
                
                return np.array([real_x, real_y, 0], dtype=np.float64)
            
            # 基于人体比例估算关键点（使用实际检测的部位中心作为参考）
            head_center = get_body_part_center(body_part_groups['head'])
            torso_center = get_body_part_center(body_part_groups['torso'])
            
            # 如果能检测到头部和躯干，使用实际位置；否则使用比例估算
            if head_center is not None and torso_center is not None:
                # 使用检测到的关键部位
                head_y = head_center[1]
                torso_y = torso_center[1]
                spine_y = (torso_y + y1 + height * 0.5) / 2
            else:
                # 使用标准比例
                head_y = y1 + height * 0.05
                torso_y = y1 + height * 0.35
                spine_y = y1 + height * 0.5
            
            keypoints = np.array([
                [center_x, y1 + height * 0.85, 0],              # 0: Hips (臀部)
                [center_x, y1 + height * 0.65, 0],              # 1: Spine (脊柱下段)
                [center_x, y1 + height * 0.45, 0],              # 2: Spine1 (脊柱上段)
                [center_x, y1 + height * 0.25, 0],              # 3: Neck (颈部)
                [center_x, head_y, 0],                          # 4: Head (头部) - 使用检测值
                [center_x + width * 0.20, y1 + height * 0.35, 0],  # 5: LeftShoulder (左肩)
                [center_x + width * 0.30, y1 + height * 0.50, 0],  # 6: LeftArm (左上臂)
                [center_x + width * 0.35, y1 + height * 0.65, 0],  # 7: LeftForeArm (左前臂)
                [center_x + width * 0.38, y1 + height * 0.80, 0],  # 8: LeftHand (左手)
                [center_x - width * 0.20, y1 + height * 0.35, 0],  # 9: RightShoulder (右肩)
                [center_x - width * 0.30, y1 + height * 0.50, 0],  # 10: RightArm (右上臂)
                [center_x - width * 0.35, y1 + height * 0.65, 0],  # 11: RightForeArm (右前臂)
                [center_x - width * 0.38, y1 + height * 0.80, 0],  # 12: RightHand (右手)
                [center_x + width * 0.12, y1 + height * 0.90, 0],  # 13: LeftUpLeg (左大腿)
                [center_x + width * 0.14, y2 - height * 0.05, 0],  # 14: LeftLeg (左小腿)
                [center_x + width * 0.15, y2, 0],                  # 15: LeftFoot (左脚)
                [center_x - width * 0.12, y1 + height * 0.90, 0],  # 16: RightUpLeg (右大腿)
                [center_x - width * 0.14, y2 - height * 0.05, 0],  # 17: RightLeg (右小腿)
                [center_x - width * 0.15, y2, 0],                  # 18: RightFoot (右脚)
            ], dtype=np.float64)
            
            return keypoints
                
        except Exception as e:
            print(f"提取关键点时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def keypoints_to_rotations(self, keypoints: np.ndarray) -> np.ndarray:
        """
        将关键点坐标转换为骨骼旋转角度
        
        Args:
            keypoints: 关键点坐标 shape: (num_joints, 3)
            
        Returns:
            旋转角度数组 shape: (num_channels,)
        """
        # 总通道数：Hips (6) + 其他所有关节 (3*18) = 60
        num_channels = 60
        rotations = np.zeros(num_channels, dtype=np.float64)
        
        # Hips位置（前3个通道是位置）
        if len(keypoints) > 0:
            rotations[0:3] = keypoints[0]  # Hips position (x, y, z)
        
        # 计算各关节的旋转
        # 简化版：计算相对于父关节的方向向量，转换为欧拉角
        channel_idx = 3
        
        # 关节父子关系索引映射
        joint_pairs = [
            (0, 1),   # Hips -> Spine
            (1, 2),   # Spine -> Spine1
            (2, 3),   # Spine1 -> Neck
            (3, 4),   # Neck -> Head
            (2, 5),   # Spine1 -> LeftShoulder
            (5, 6),   # LeftShoulder -> LeftArm
            (6, 7),   # LeftArm -> LeftForeArm
            (7, 8),   # LeftForeArm -> LeftHand
            (2, 9),   # Spine1 -> RightShoulder
            (9, 10),  # RightShoulder -> RightArm
            (10, 11), # RightArm -> RightForeArm
            (11, 12), # RightForeArm -> RightHand
            (0, 13),  # Hips -> LeftUpLeg
            (13, 14), # LeftUpLeg -> LeftLeg
            (14, 15), # LeftLeg -> LeftFoot
            (0, 16),  # Hips -> RightUpLeg
            (16, 17), # RightUpLeg -> RightLeg
            (17, 18), # RightLeg -> RightFoot
        ]
        
        for parent_idx, child_idx in joint_pairs:
            if child_idx < len(keypoints) and parent_idx < len(keypoints):
                # 计算方向向量
                direction = keypoints[child_idx] - keypoints[parent_idx]
                
                # 计算旋转角度（简化的欧拉角）
                # Z rotation (绕Z轴旋转 - 左右摆动)
                z_rot = np.degrees(np.arctan2(direction[0], direction[1]))
                
                # X rotation (绕X轴旋转 - 前后倾斜)
                xy_dist = np.sqrt(direction[0]**2 + direction[1]**2)
                x_rot = np.degrees(np.arctan2(direction[2], xy_dist))
                
                # Y rotation (绕Y轴旋转 - 扭转)
                y_rot = 0.0  # 简化处理，2D数据无法准确估计扭转
                
                if channel_idx + 2 < num_channels:
                    rotations[channel_idx] = z_rot
                    rotations[channel_idx + 1] = x_rot
                    rotations[channel_idx + 2] = y_rot
                    channel_idx += 3
                
        return rotations
    
    def write_hierarchy(self, f, node_name: str, node_data: dict, indent: int = 0):
        """递归写入骨骼层次结构"""
        tab = "  " * indent
        
        # 写入节点名称
        if indent == 0:
            f.write(f"{tab}ROOT {node_name}\n")
        else:
            f.write(f"{tab}JOINT {node_name}\n")
        
        f.write(f"{tab}{{\n")
        
        # 写入偏移
        offset = node_data["offset"]
        f.write(f"{tab}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # 写入通道
        channels = node_data["channels"]
        f.write(f"{tab}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # 递归写入子节点
        children = node_data.get("children", {})
        if children:
            for child_name, child_data in children.items():
                self.write_hierarchy(f, child_name, child_data, indent + 1)
        else:
            # 末端节点
            f.write(f"{tab}  End Site\n")
            f.write(f"{tab}  {{\n")
            f.write(f"{tab}    OFFSET 0.000000 0.000000 0.000000\n")
            f.write(f"{tab}  }}\n")
        
        f.write(f"{tab}}}\n")
    
    def export_to_bvh(
        self,
        keypoints_sequence: List[Optional[np.ndarray]],
        output_path: str,
        scale: float = 1.0
    ) -> bool:
        """
        导出BVH文件
        
        Args:
            keypoints_sequence: 每帧的关键点序列
            output_path: 输出文件路径
            scale: 缩放因子
            
        Returns:
            是否成功导出
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                # 写入HIERARCHY部分
                f.write("HIERARCHY\n")
                self.write_hierarchy(f, "Hips", self.skeleton.HIERARCHY["Hips"])
                
                # 写入MOTION部分
                f.write("MOTION\n")
                f.write(f"Frames: {len(keypoints_sequence)}\n")
                f.write(f"Frame Time: {self.frame_time:.6f}\n")
                
                # 统计有效帧数用于调试
                valid_frames = sum(1 for kp in keypoints_sequence if kp is not None)
                print(f"生成 BVH: 总帧数 {len(keypoints_sequence)}, 有效帧 {valid_frames}")
                
                # 写入每帧的运动数据
                for frame_idx, frame_keypoints in enumerate(keypoints_sequence):
                    if frame_keypoints is not None:
                        # 缩放关键点
                        scaled_keypoints = frame_keypoints * scale
                        # 转换为旋转
                        rotations = self.keypoints_to_rotations(scaled_keypoints)
                    else:
                        # 如果该帧没有检测到人体，使用零姿态
                        rotations = np.zeros(60, dtype=np.float64)
                    
                    # 写入旋转数据
                    rotation_str = ' '.join([f"{r:.6f}" for r in rotations])
                    f.write(f"{rotation_str}\n")
                    
                    # 每10帧打印一次进度（用于调试）
                    if frame_idx % 10 == 0 and frame_keypoints is not None:
                        print(f"  帧 {frame_idx}: 位置 ({rotations[0]:.2f}, {rotations[1]:.2f}, {rotations[2]:.2f})")
            
            print(f"✓ BVH文件已导出: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ 导出BVH文件失败: {e}")
            return False


def create_bvh_from_densepose_results(
    densepose_results: List,
    output_path: str,
    fps: float = 30.0,
    scale: float = 0.1
) -> bool:
    """
    从DensePose结果序列创建BVH文件
    
    Args:
        densepose_results: DensePose检测结果列表
        output_path: 输出BVH文件路径
        fps: 帧率
        scale: 坐标缩放因子
        
    Returns:
        是否成功创建
    """
    exporter = BVHExporter(fps=fps)
    
    # 提取所有帧的关键点
    keypoints_sequence = []
    for result in densepose_results:
        keypoints = exporter.extract_keypoints_from_densepose(result)
        keypoints_sequence.append(keypoints)
    
    # 导出BVH
    return exporter.export_to_bvh(keypoints_sequence, output_path, scale=scale)
