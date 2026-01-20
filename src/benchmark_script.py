# Create a simple benchmark script
import time
import torch
import torch.backends.cudnn as cudnn
from lib.models.model import create_model, load_model
from lib.opts import opts
import numpy as np

cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Benchmark:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def force_compilation(self, model, input_tensor, iterations=20):
        """
        Force all JIT compilation to happen before timing.
        This is critical for accurate benchmarking.
        """
        print("Forcing model compilation...")
        
        with torch.no_grad():
            for i in range(iterations):
                _ = model(input_tensor)
                if (i + 1) % 5 == 0:
                    print(f"  Compilation iteration {i+1}/{iterations}")
        
        torch.cuda.synchronize()
        print("Compilation complete!\n")
    def run_benchmark(self, iterations, batch_size=1):
      
        self.model = self.model.cuda().eval()
        input_tensor = torch.randn(batch_size, 3, 512, 512).cuda()
        # self.force_compilation(self.model, input_tensor, iterations=20)

        #Warmup
        for i in range(50):
            with torch.no_grad():
                _ = self.model(input_tensor)

        torch.cuda.synchronize()
        
        # Clear cache to start fresh
        torch.cuda.empty_cache()
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for i in range(iterations):
            # torch.cuda.synchronize()
            start_event.record()
        
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            # Record end on GPU
            end_event.record()
            
            # Wait for GPU to finish
            torch.cuda.synchronize()
            
            # Get elapsed time (in milliseconds)
            elapsed_ms = start_event.elapsed_time(end_event)
            times_ms.append(elapsed_ms)
            
            if (i+1)%50 == 0:
                print(f"{i+1} iterations complete")
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        # times_ms = times_ms[10:] #Trim times as first might be unstable
        times_trimmed = times_ms[10:]
    
        avg_time = np.mean(times_trimmed)
        std_time = np.std(times_trimmed)
        min_time = np.min(times_trimmed)
        max_time = np.max(times_trimmed)

        print(f"\nResults:")
        print(f"  Average:   {avg_time:.2f} ms")
        print(f"  Std Dev:   {std_time:.2f} ms")
        print(f"  Min:       {min_time:.2f} ms")
        print(f"  Max:       {max_time:.2f} ms")
        print(f"  Rel. Std:  {(std_time/avg_time)*100:.1f}%")
        print(f"  FPS:       {batch_size / (avg_time/1000):.2f}")
        
        if (std_time / avg_time) > 0.05:  # >5% is concerning
            print(f"  ⚠️  WARNING: High variance!")
        
        return {
            'batch_size': batch_size,
            'avg_ms': avg_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'fps': batch_size / (avg_time/1000)
        }
        

        
def main():
    opt = opts().parse()
    opt.arch = 'dla_34'
    opt.heads = {'hm': 20, 'wh': 2, 'reg': 2}
    opt.head_conv = 256

    model = create_model(opt.arch, opt.heads, opt.head_conv) #Head conv = 256
    model = load_model(model, '/home/orin/Dev/CenterNet/CenterNet-edge-testing/models/ctdet_pascal_dla_384.pth')
    model = model.cuda().eval()
    device = "cuda"
    num_iterations = 100

    DLA34_benchmark = Benchmark(model=model, device=device)
    
    for batch_size in [1,2,4,8]:
        print(f"Batch size {batch_size}:\n")
        _= DLA34_benchmark.run_benchmark(num_iterations, batch_size=batch_size)
        # print(f"Average = {avg:.2f}, Std = {std:.2f}\n")
        # time.sleep(2)

if __name__ == "__main__":
    main()