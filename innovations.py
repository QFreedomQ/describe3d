"""
创新模块：包含三个主要创新点的实现

创新点1：多视角渲染和一致性优化
创新点2：渐进式优化策略
创新点3：质量评估和自动保存最佳结果
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex
    )
    PYTORCH3D_AVAILABLE = True
except ImportError as e:
    PYTORCH3D_AVAILABLE = False
    print(f"Warning: PyTorch3D not available ({e}). Multi-view rendering will be limited.")


class MultiViewRenderer:
    """
    创新点1：多视角渲染器
    支持从多个角度渲染3D人脸，增强生成质量和一致性
    """
    
    def __init__(self, device="cuda", image_size=512):
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D is required for MultiViewRenderer. Please install it first.")
        
        self.device = device
        self.image_size = image_size
        
        # 定义多个视角：前、左、右、左上、右上
        self.view_angles = {
            'front': {'elev': 0, 'azim': 0},
            'left': {'elev': 0, 'azim': -30},
            'right': {'elev': 0, 'azim': 30},
            'top_left': {'elev': 15, 'azim': -20},
            'top_right': {'elev': 15, 'azim': 20},
        }
        
    def render_multi_view(self, curr_verts, render_img, view_name='front'):
        """
        从指定视角渲染人脸
        
        Args:
            curr_verts: 当前顶点位置
            render_img: 纹理图像
            view_name: 视角名称
        
        Returns:
            渲染后的图像
        """
        mean_verts = torch.from_numpy(np.load("./predef/mean_verts.npy")).cuda()
        faces = torch.from_numpy(np.load("./predef/faces.npy")).to(torch.int64).cuda()
        
        vertices = (curr_verts + mean_verts).unsqueeze(0)
        faces_tensor = faces.unsqueeze(0)
        
        num_verts = vertices.shape[1]
        texture_rgb = (render_img[0] + 1) / 2
        texture_rgb = texture_rgb.permute(1, 2, 0).reshape(-1, 3)
        
        if texture_rgb.shape[0] < num_verts:
            texture_rgb = torch.nn.functional.interpolate(
                texture_rgb.unsqueeze(0).permute(0, 2, 1),
                size=num_verts,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1).squeeze(0)
        elif texture_rgb.shape[0] > num_verts:
            indices = torch.linspace(0, texture_rgb.shape[0] - 1, num_verts).long()
            texture_rgb = texture_rgb[indices]
        
        verts_rgb = texture_rgb.unsqueeze(0)
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(verts=vertices, faces=faces_tensor, textures=textures)
        
        # 根据视角设置相机
        view_params = self.view_angles.get(view_name, self.view_angles['front'])
        R, T = look_at_view_transform(
            dist=2.7, 
            elev=view_params['elev'], 
            azim=view_params['azim']
        )
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        
        images = renderer(mesh)
        img_pred = images[0, ..., :3]
        
        return img_pred
    
    def compute_multi_view_consistency_loss(self, curr_verts, render_img):
        """
        计算多视角一致性损失
        通过比较不同视角渲染结果的特征相似性来确保3D一致性
        """
        views = ['front', 'left', 'right']
        rendered_views = []
        
        for view_name in views:
            img = self.render_multi_view(curr_verts, render_img, view_name)
            rendered_views.append(img)
        
        # 计算视角间的特征一致性
        consistency_loss = 0.0
        for i in range(len(rendered_views)):
            for j in range(i + 1, len(rendered_views)):
                # 使用L2距离衡量一致性（在特征空间）
                view_i = rendered_views[i].mean(dim=[0, 1])  # 平均特征
                view_j = rendered_views[j].mean(dim=[0, 1])
                consistency_loss += torch.nn.functional.mse_loss(view_i, view_j)
        
        return consistency_loss / len(rendered_views)


class ProgressiveOptimizer:
    """
    创新点2：渐进式优化策略
    分阶段优化纹理和形状，每个阶段有不同的学习率和权重
    """
    
    def __init__(self, total_steps, initial_lr_latent=0.008, initial_lr_param=0.003,
                 initial_lambda_latent=0.0003, initial_lambda_param=3.0):
        self.total_steps = total_steps
        self.initial_lr_latent = initial_lr_latent
        self.initial_lr_param = initial_lr_param
        self.initial_lambda_latent = initial_lambda_latent
        self.initial_lambda_param = initial_lambda_param
        
        # 定义三个阶段
        self.stage1_end = int(total_steps * 0.4)  # 前40%：重点优化纹理
        self.stage2_end = int(total_steps * 0.7)  # 40-70%：重点优化形状
        # stage3: 70-100%：联合精细化
        
    def get_current_params(self, step):
        """
        根据当前步数返回学习率和正则化权重
        """
        if step < self.stage1_end:
            # 阶段1：重点优化纹理
            lr_latent = self.initial_lr_latent * 1.5  # 提高纹理学习率
            lr_param = self.initial_lr_param * 0.5     # 降低形状学习率
            lambda_latent = self.initial_lambda_latent * 0.5  # 降低纹理正则化
            lambda_param = self.initial_lambda_param * 1.5    # 提高形状正则化
            stage = "Stage 1: Texture Focus"
            
        elif step < self.stage2_end:
            # 阶段2：重点优化形状
            lr_latent = self.initial_lr_latent * 0.5
            lr_param = self.initial_lr_param * 1.5
            lambda_latent = self.initial_lambda_latent * 1.5
            lambda_param = self.initial_lambda_param * 0.5
            stage = "Stage 2: Shape Focus"
            
        else:
            # 阶段3：联合精细化
            # 逐渐降低学习率
            progress = (step - self.stage2_end) / (self.total_steps - self.stage2_end)
            decay = 1.0 - 0.5 * progress
            lr_latent = self.initial_lr_latent * decay
            lr_param = self.initial_lr_param * decay
            lambda_latent = self.initial_lambda_latent
            lambda_param = self.initial_lambda_param
            stage = "Stage 3: Joint Refinement"
        
        return {
            'lr_latent': lr_latent,
            'lr_param': lr_param,
            'lambda_latent': lambda_latent,
            'lambda_param': lambda_param,
            'stage': stage
        }


class QualityEvaluator:
    """
    创新点3：质量评估器
    评估每次迭代的结果质量，自动保存最佳结果
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_score = float('inf')
        self.best_iteration = 0
        self.history = {
            'iteration': [],
            'clip_loss': [],
            'l2_latent': [],
            'l2_param': [],
            'total_loss': [],
            'quality_score': []
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate(self, iteration, clip_loss, l2_latent, l2_param, total_loss):
        """
        评估当前迭代的质量
        
        质量分数综合考虑：
        - CLIP损失（越小越好）
        - L2正则化（过大或过小都不好，需要平衡）
        - 总损失
        """
        # 归一化各个指标
        clip_weight = 0.6
        regularization_weight = 0.4
        
        # CLIP损失是主要指标
        clip_score = clip_loss.item() * clip_weight
        
        # 正则化分数：希望在合理范围内（不要过度偏离初始值）
        reg_score = (l2_latent.item() + l2_param.item()) * regularization_weight
        
        # 综合质量分数（越小越好）
        quality_score = clip_score + reg_score * 0.1
        
        # 记录历史
        self.history['iteration'].append(iteration)
        self.history['clip_loss'].append(clip_loss.item())
        self.history['l2_latent'].append(l2_latent.item())
        self.history['l2_param'].append(l2_param.item())
        self.history['total_loss'].append(total_loss.item())
        self.history['quality_score'].append(quality_score)
        
        # 判断是否是最佳结果
        is_best = quality_score < self.best_score
        if is_best:
            self.best_score = quality_score
            self.best_iteration = iteration
        
        return is_best, quality_score
    
    def save_best_state(self, latent, param, iteration):
        """保存最佳状态"""
        best_state = {
            'latent': latent.detach().cpu(),
            'param': param.detach().cpu(),
            'iteration': iteration,
            'score': self.best_score
        }
        torch.save(best_state, os.path.join(self.save_dir, 'best_model.pth'))
    
    def generate_report(self):
        """
        生成优化过程的可视化报告
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制CLIP损失曲线
        axes[0, 0].plot(self.history['iteration'], self.history['clip_loss'], 'b-', linewidth=2)
        axes[0, 0].axvline(x=self.best_iteration, color='r', linestyle='--', label='Best')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('CLIP Loss')
        axes[0, 0].set_title('CLIP Loss over Iterations')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 绘制L2正则化曲线
        axes[0, 1].plot(self.history['iteration'], self.history['l2_latent'], 
                       'g-', label='L2 Latent', linewidth=2)
        axes[0, 1].plot(self.history['iteration'], self.history['l2_param'], 
                       'orange', label='L2 Param', linewidth=2)
        axes[0, 1].axvline(x=self.best_iteration, color='r', linestyle='--', label='Best')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('L2 Regularization')
        axes[0, 1].set_title('Regularization over Iterations')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 绘制总损失曲线
        axes[1, 0].plot(self.history['iteration'], self.history['total_loss'], 
                       'purple', linewidth=2)
        axes[1, 0].axvline(x=self.best_iteration, color='r', linestyle='--', label='Best')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].set_title('Total Loss over Iterations')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 绘制质量分数曲线
        axes[1, 1].plot(self.history['iteration'], self.history['quality_score'], 
                       'red', linewidth=2)
        axes[1, 1].axvline(x=self.best_iteration, color='r', linestyle='--', label='Best')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Quality Score over Iterations (Lower is Better)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'optimization_report.png'), dpi=150)
        plt.close()
        
        # 保存JSON报告
        report = {
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'final_metrics': {
                'clip_loss': self.history['clip_loss'][-1],
                'l2_latent': self.history['l2_latent'][-1],
                'l2_param': self.history['l2_param'][-1],
                'total_loss': self.history['total_loss'][-1]
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.save_dir, 'optimization_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
