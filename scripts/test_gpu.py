import torch
import time

def test():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        
        # Test basic tensor op
        print("Testing GPU matrix multiplication...")
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Matrix multiplication done in {end-start:.4f}s")
        
        # Test memory allocation
        print("Testing 1GB allocation...")
        mem = torch.randn(1024, 1024, 256).cuda() # ~1GB
        torch.cuda.synchronize()
        print("Allocation success.")
    else:
        print("CUDA not available.")

if __name__ == "__main__":
    test()
