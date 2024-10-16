import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available! GPU is ready to use.")
    # Print the name of the GPU device
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")


def get_gpu_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # 假设使用第一块 GPU
        props = torch.cuda.get_device_properties(device)

        total_memory = props.total_memory / (1024 ** 3)  # 转换为 GB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # 当前分配的显存
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # 当前预留的显存
        available_memory = total_memory - reserved_memory

        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory:.2f} GB")
        print(f"Reserved GPU memory: {reserved_memory:.2f} GB")
        print(f"Available GPU memory: {available_memory:.2f} GB")


get_gpu_memory()
