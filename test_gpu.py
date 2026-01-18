import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU device:', torch.cuda.get_device_name(0))
print('GPU count:', torch.cuda.device_count())
print('Supported architectures:', torch.cuda.get_arch_list())   