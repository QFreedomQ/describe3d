#!/usr/bin/env python3
"""Test script to verify CUDA plugin fallback mechanism."""

import torch
import sys

print("Testing plugin fallback mechanism...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Test bias_act
print("\n=== Testing bias_act ===")
try:
    from torch_utils.ops import bias_act
    x = torch.randn(1, 10, 32, 32)
    if torch.cuda.is_available():
        x = x.cuda()
    y = bias_act.bias_act(x, act='lrelu')
    print(f"✓ bias_act works! Output shape: {y.shape}")
except Exception as e:
    print(f"✗ bias_act failed: {e}")
    sys.exit(1)

# Test upfirdn2d
print("\n=== Testing upfirdn2d ===")
try:
    from torch_utils.ops import upfirdn2d
    x = torch.randn(1, 3, 32, 32)
    if torch.cuda.is_available():
        x = x.cuda()
    f = upfirdn2d.setup_filter([1, 3, 3, 1], device=x.device)
    y = upfirdn2d.upfirdn2d(x, f, up=2, down=1)
    print(f"✓ upfirdn2d works! Output shape: {y.shape}")
except Exception as e:
    print(f"✗ upfirdn2d failed: {e}")
    sys.exit(1)

# Test filtered_lrelu
print("\n=== Testing filtered_lrelu ===")
try:
    from torch_utils.ops import filtered_lrelu
    x = torch.randn(1, 3, 32, 32)
    if torch.cuda.is_available():
        x = x.cuda()
    y = filtered_lrelu.filtered_lrelu(x)
    print(f"✓ filtered_lrelu works! Output shape: {y.shape}")
except Exception as e:
    print(f"✗ filtered_lrelu failed: {e}")
    sys.exit(1)

print("\n=== All tests passed! ===")
