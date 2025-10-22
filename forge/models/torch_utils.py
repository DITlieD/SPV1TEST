# forge/models/torch_utils.py
import torch

def get_device():
    """Determines the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"PyTorch initialized using device: {DEVICE}")