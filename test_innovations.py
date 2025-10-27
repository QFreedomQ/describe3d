#!/usr/bin/env python3
"""
测试脚本：验证创新功能的正确性

这个脚本测试三个创新模块是否能正确导入和初始化，
不需要预训练模型或GPU环境。
"""

import sys
import os
import torch
import numpy as np

def test_import():
    """测试模块导入"""
    print("=" * 70)
    print("测试1: 模块导入")
    print("=" * 70)
    
    try:
        from innovations import MultiViewRenderer, ProgressiveOptimizer, QualityEvaluator
        print("✓ 成功导入 MultiViewRenderer")
        print("✓ 成功导入 ProgressiveOptimizer")
        print("✓ 成功导入 QualityEvaluator")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_progressive_optimizer():
    """测试渐进式优化器"""
    print("\n" + "=" * 70)
    print("测试2: 渐进式优化器")
    print("=" * 70)
    
    try:
        from innovations import ProgressiveOptimizer
        
        opt = ProgressiveOptimizer(
            total_steps=100,
            initial_lr_latent=0.008,
            initial_lr_param=0.003,
            initial_lambda_latent=0.0003,
            initial_lambda_param=3.0
        )
        
        # 测试不同阶段的参数
        stages_to_test = [0, 40, 70, 99]
        print("\n测试不同优化阶段的参数调整:")
        
        for step in stages_to_test:
            params = opt.get_current_params(step)
            print(f"\n  步骤 {step}:")
            print(f"    阶段: {params['stage']}")
            print(f"    lr_latent: {params['lr_latent']:.6f}")
            print(f"    lr_param: {params['lr_param']:.6f}")
            print(f"    lambda_latent: {params['lambda_latent']:.6f}")
            print(f"    lambda_param: {params['lambda_param']:.2f}")
        
        print("\n✓ 渐进式优化器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_evaluator():
    """测试质量评估器"""
    print("\n" + "=" * 70)
    print("测试3: 质量评估器")
    print("=" * 70)
    
    try:
        from innovations import QualityEvaluator
        import tempfile
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"\n使用临时目录: {temp_dir}")
        
        evaluator = QualityEvaluator(save_dir=temp_dir)
        
        # 模拟优化过程
        print("\n模拟优化过程（10次迭代）:")
        best_iter = -1
        
        for i in range(10):
            # 模拟损失值（逐渐下降）
            clip_loss = torch.tensor(0.5 - i * 0.03)
            l2_latent = torch.tensor(2.0 + i * 0.1)
            l2_param = torch.tensor(1.5 + i * 0.05)
            total_loss = clip_loss + l2_latent * 0.0003 + l2_param * 3
            
            is_best, quality_score = evaluator.evaluate(
                i, clip_loss, l2_latent, l2_param, total_loss
            )
            
            if is_best:
                best_iter = i
                print(f"  迭代 {i}: 质量分数 {quality_score:.4f} ⭐ (最佳)")
                # 保存模拟的最佳状态
                latent = torch.randn(1, 512)
                param = torch.randn(1, 300)
                evaluator.save_best_state(latent, param, i)
            else:
                print(f"  迭代 {i}: 质量分数 {quality_score:.4f}")
        
        # 生成报告
        print("\n生成优化报告...")
        report = evaluator.generate_report()
        
        print(f"\n最佳迭代: {report['best_iteration']}")
        print(f"最佳质量分数: {report['best_score']:.4f}")
        
        # 检查生成的文件
        report_files = [
            os.path.join(temp_dir, 'optimization_report.png'),
            os.path.join(temp_dir, 'optimization_report.json'),
            os.path.join(temp_dir, 'best_model.pth')
        ]
        
        print("\n检查生成的文件:")
        all_exist = True
        for f in report_files:
            exists = os.path.exists(f)
            status = "✓" if exists else "✗"
            print(f"  {status} {os.path.basename(f)}")
            all_exist = all_exist and exists
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n已清理临时目录")
        
        if all_exist:
            print("\n✓ 质量评估器测试通过")
            return True
        else:
            print("\n✗ 部分文件未生成")
            return False
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_view_renderer_init():
    """测试多视角渲染器初始化"""
    print("\n" + "=" * 70)
    print("测试4: 多视角渲染器初始化")
    print("=" * 70)
    
    try:
        from innovations import MultiViewRenderer
        
        # 测试CPU初始化（不需要GPU）
        renderer = MultiViewRenderer(device="cpu", image_size=256)
        
        print("\n视角配置:")
        for view_name, params in renderer.view_angles.items():
            print(f"  {view_name}: elev={params['elev']}°, azim={params['azim']}°")
        
        print("\n✓ 多视角渲染器初始化成功")
        print("  注意: 完整的渲染测试需要预训练模型和GPU环境")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  创新功能单元测试")
    print("=" * 70)
    print("\n本测试不需要预训练模型或GPU，只验证代码逻辑的正确性。\n")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_import()))
    results.append(("渐进式优化器", test_progressive_optimizer()))
    results.append(("质量评估器", test_quality_evaluator()))
    results.append(("多视角渲染器", test_multi_view_renderer_init()))
    
    # 总结
    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
    
    if passed == total:
        print("\n🎉 所有测试通过！创新功能已正确实现。")
        print("\n下一步:")
        print("  1. 准备预训练模型和必需文件")
        print("  2. 运行 python demo_innovations.py 查看使用示例")
        print("  3. 使用 python main.py 进行实际的3D人脸生成")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查代码。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
