"""
Test script for RTX 50 series GPU compatibility with PyTorch
Run this to verify your GPU is properly recognized and accessible
"""

# Try to import compatibility layer if available
try:
    import rtx50_compat
    print("✓ rtx50-compat compatibility layer loaded")
except ImportError:
    print("ℹ rtx50-compat not installed (optional)")

import torch

print('\n' + '='*50)
print('PyTorch GPU Compatibility Test')
print('='*50)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')

if torch.cuda.is_available():
    print('\nGPU Information:')
    print('  GPU device:', torch.cuda.get_device_name(0))
    print('  GPU count:', torch.cuda.device_count())
    print('  Compute capability:', torch.cuda.get_device_capability(0))
    print('  Supported architectures:', torch.cuda.get_arch_list())
    
    # Test basic tensor operations
    print('\nTesting GPU operations:')
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print('  ✓ GPU tensor operations working')
        print('  ✓ Memory allocated:', f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        print('  ✗ GPU operations failed:', str(e))
else:
    print('\n⚠ CUDA not available - GPU not detected or drivers not installed')
    print('  Check:')
    print('    1. NVIDIA drivers are installed')
    print('    2. CUDA toolkit is installed')
    print('    3. PyTorch was built with CUDA support')
    print('    4. For RTX 50 series: consider using rtx50-compat or stone-linux')

print('='*50)   