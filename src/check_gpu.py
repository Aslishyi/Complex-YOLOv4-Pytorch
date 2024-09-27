import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available! GPU is ready to use.")
    # Print the name of the GPU device
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")
