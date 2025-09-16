# check_torch_cuda.py
import time
import torch
import sys

def main():
    print("Python:", sys.version.splitlines()[0])
    print("torch:", torch.__version__)
    try:
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
    except Exception as e:
        print("Lỗi khi truy vấn thông tin CUDA:", e)

    try:
        print("cuDNN enabled:", torch.backends.cudnn.enabled)
    except Exception:
        pass

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print("CUDA device count:", n_gpus)
        for i in range(n_gpus):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        cur = torch.cuda.current_device()
        print("Current device index:", cur)
        print("Current device name:", torch.cuda.get_device_name(cur))

        # Thử 1 phép toán lớn trên GPU để kiểm tra thực tế
        print("\nThực hiện test matmul trên GPU (1000x1000)...")
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        # đồng bộ trước khi đo
        torch.cuda.synchronize()
        t0 = time.time()
        c = torch.matmul(a, b)
        # đồng bộ sau phép tính để thời gian chính xác
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Matmul 1000x1000 trên GPU mất: {(t1 - t0):.4f} giây")

        # kiểm tra kết quả chuyển về CPU
        s = c.sum().item()
        print("Sum của kết quả (để kiểm tra):", s)
    else:
        print("\nCUDA không khả dụng. Thử in thông tin driver/NVIDIA:")
        # Thử import nvidia-smi nếu có (không bắt buộc)
        try:
            import subprocess
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            print("--- nvidia-smi output ---")
            print(out)
        except Exception as e:
            print("Không thể chạy nvidia-smi hoặc không có GPU / driver:", e)

if __name__ == "__main__":
    main()
