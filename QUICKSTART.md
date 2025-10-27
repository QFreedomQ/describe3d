# 快速开始指南

本指南帮助你在5分钟内开始使用创新版的Describe3D项目。

---

## 第一步：环境准备

### 1.1 检查系统要求
- ✅ Python 3.8
- ✅ CUDA 11.0+
- ✅ GPU（至少8GB显存推荐）

### 1.2 安装依赖
```bash
# 创建虚拟环境
conda create -n describe3d python=3.8
conda activate describe3d

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git

# 安装其他依赖
pip install -r requirements.txt
```

---

## 第二步：下载模型

从 [Google Drive](https://drive.google.com/drive/folders/1zqCLaF-KzhWy_YSMqKf15aEKiv19lXz5?usp=sharing) 下载：

1. `latest_texture.pkl` → 放到 `checkpoints/texture_synthesis/`
2. `latest_shape.pth` → 放到 `checkpoints/shape_synthesis/`
3. `latest_parser.pth` → 放到 `checkpoints/onehot_classfier/`

确保文件结构如下：
```
checkpoints/
├── texture_synthesis/
│   └── latest_texture.pkl
├── shape_synthesis/
│   └── latest_shape.pth
└── onehot_classfier/
    └── latest_parser.pth
```

---

## 第三步：运行第一个示例

### 3.1 基础示例（原始功能）
```bash
python main.py \
    --name="my_first_face" \
    --descriptions="A young woman with blue eyes and blonde hair." \
    --prompt="beautiful woman" \
    --step=50
```

**预期时间**：约5-10分钟（取决于GPU）  
**输出**：`result/final_result/my_first_face/`

---

### 3.2 高质量示例（使用创新功能）
```bash
python main.py \
    --name="high_quality_face" \
    --descriptions="A young woman with blue eyes and blonde hair. She has a round face and a small nose." \
    --prompt="beautiful woman" \
    --use_multi_view \
    --save_multi_view \
    --step=100
```

**预期时间**：约10-15分钟  
**输出**：
- 3D模型（.obj文件）
- 5个视角的渲染图像
- 优化报告（图表+数值）

---

## 第四步：查看结果

### 4.1 3D模型
使用任何3D查看器打开 `.obj` 文件：
- **Windows**: 3D Viewer, Blender
- **Mac**: Preview, Blender
- **Linux**: Blender, MeshLab
- **在线**: [3D Viewer Online](https://3dviewer.net/)

主要文件：
```
result/final_result/[name]/[prompt]/result_prompt.obj
```

### 4.2 多视角渲染
如果使用了 `--save_multi_view`，可以查看：
```
result/final_result/[name]/[prompt]/
├── view_front.jpg      # 正面
├── view_left.jpg       # 左侧
├── view_right.jpg      # 右侧
├── view_top_left.jpg   # 左上
└── view_top_right.jpg  # 右上
```

### 4.3 优化报告
```
optimization_report.png  # 包含4个子图的优化曲线
optimization_report.json # 数值分析结果
```

---

## 第五步：尝试更多示例

### 示例1：生成男性人脸
```bash
python main.py \
    --name="man_face" \
    --descriptions="A middle-aged man with a beard and brown eyes. He has a strong jawline and short hair." \
    --prompt="handsome man" \
    --use_multi_view \
    --step=100
```

### 示例2：生成老年人脸
```bash
python main.py \
    --name="elderly_face" \
    --descriptions="An elderly woman with wrinkles and gray hair. She has a kind smile and gentle eyes." \
    --prompt="grandmother" \
    --use_multi_view \
    --step=100
```

### 示例3：生成名人脸（Tony Stark）
```bash
python main.py \
    --name="Tony_Stark" \
    --descriptions="This middle-aged man is a westerner. He has big and black eyes with the double eyelid. He has a medium-sized nose with a high nose bridge. His face is square and medium. He has a dense and black beard." \
    --prompt="Tony Stark" \
    --use_multi_view \
    --save_multi_view \
    --step=150 \
    --lambda_latent=0.0003 \
    --lambda_param=3
```

---

## 常用参数组合

### 快速测试（低质量，快速）
```bash
--step=50
# 不使用 --use_multi_view
```
⏱️ 时间：5分钟  
🎨 质量：中等

---

### 标准生成（平衡质量和速度）
```bash
--use_multi_view --step=100
```
⏱️ 时间：10分钟  
🎨 质量：良好

---

### 高质量生成（推荐）
```bash
--use_multi_view --save_multi_view --step=150
```
⏱️ 时间：15分钟  
🎨 质量：优秀

---

### 研究级生成（最高质量）
```bash
--use_multi_view --save_multi_view --step=200 --save_step=10
```
⏱️ 时间：20分钟  
🎨 质量：最佳  
📊 输出：包含完整中间结果

---

## 故障排除

### 问题1：CUDA内存不足
**错误信息**：`RuntimeError: CUDA out of memory`

**解决方案**：
1. 减少迭代步数：`--step=50`
2. 不使用多视角：移除 `--use_multi_view`
3. 使用更小的图像尺寸（需修改代码）

---

### 问题2：模型文件未找到
**错误信息**：`FileNotFoundError: ... .pkl not found`

**解决方案**：
1. 确认模型文件已下载
2. 检查文件路径是否正确
3. 使用 `--TextureNet_path` 等参数指定路径

---

### 问题3：PyTorch3D导入失败
**错误信息**：`ImportError: libcudart.so.10.1 not found`

**解决方案**：
```bash
# 重新安装PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 或者使用conda
conda install -c facebookresearch -c iopath -c conda-forge pytorch3d
```

---

### 问题4：生成结果质量不佳
**症状**：模型形状或纹理不理想

**解决方案**：
1. 增加迭代步数：`--step=150` 或 `--step=200`
2. 启用多视角优化：`--use_multi_view`
3. 调整正则化参数：
   - 想要更大形状变化：减小 `--lambda_param`（如改为1.5）
   - 想要更大纹理变化：减小 `--lambda_latent`（如改为0.0001）
4. 改进文本描述（更详细、更准确）

---

## 查看演示和测试

### 交互式演示
```bash
python demo_innovations.py
```
这会显示各种使用示例和参数说明。

### 运行单元测试
```bash
python test_innovations.py
```
验证创新模块是否正确安装。

---

## 下一步

1. 📖 **阅读完整文档**
   - [INNOVATIONS_SUMMARY_CN.md](INNOVATIONS_SUMMARY_CN.md) - 技术细节
   - [README_CN.md](README_CN.md) - 完整说明

2. 🎨 **尝试不同描述**
   - 实验各种人脸特征描述
   - 调整提示词获得不同风格

3. 🔧 **调整参数**
   - 尝试不同的学习率
   - 调整正则化权重
   - 实验不同的迭代步数

4. 📊 **分析结果**
   - 查看优化报告理解训练过程
   - 比较不同参数的效果
   - 保存最佳配置用于未来生成

---

## 性能参考

基于RTX 3090 GPU的测试结果：

| 配置 | 时间 | GPU内存 | 质量 |
|-----|------|--------|------|
| 快速（step=50） | ~5分钟 | ~6GB | 中等 |
| 标准（step=100） | ~10分钟 | ~7GB | 良好 |
| 高质量（step=150 + multi-view） | ~15分钟 | ~8GB | 优秀 |
| 研究级（step=200 + all） | ~20分钟 | ~9GB | 最佳 |

*注：时间和内存使用会因硬件而异*

---

## 获取帮助

- 📖 阅读 [FAQ](INNOVATIONS_SUMMARY_CN.md#常见问题)
- 🐛 提交 [GitHub Issue](https://github.com/...)
- 💬 查看项目讨论区
- 📧 联系项目维护者

---

## 总结

恭喜！你已经学会了：
- ✅ 安装和配置环境
- ✅ 运行基础和高级示例
- ✅ 查看和分析结果
- ✅ 使用创新功能提升质量
- ✅ 故障排除和优化

现在你可以：
1. 生成自己的3D人脸
2. 探索不同的参数组合
3. 分析优化过程
4. 获得高质量结果

祝你使用愉快！🎉

---

**文档版本**：1.0  
**最后更新**：2024-01-15
