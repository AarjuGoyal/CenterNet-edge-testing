# Create a simple benchmark script
import time
import torch
import torch.backends.cudnn as cudnn
from lib.models.model import create_model, load_model
from collections import defaultdict
from lib.opts import opts
import numpy as np

cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Benchmark:
    def __init__(self, model, device, run_with_hooks = False):
        self.model = model
        self.device = device
        self.run_with_hooks = run_with_hooks
        if run_with_hooks == True:
            self.layer_wise_profile = True
            self.hooks = []
            self.layer_times = defaultdict(list)
            self.start_events = {}
            self.end_events = {}
            self.register_cuda_hooks()

    def warmup(self, model, input_tensor, iterations=20):
        """
        Force all JIT compilation to happen before timing.
        This is critical for accurate benchmarking.
        """
        print("Warm up..")
        
        with torch.no_grad():
            for i in range(iterations):
                _ = model(input_tensor)
                # if (i + 1) % 5 == 0:
                #     print(f"  Compilation iteration {i+1}/{iterations}")
        torch.cuda.synchronize()
    
    def register_cuda_hooks(self):

        def make_forward_pre_hook(name):
            """Hook that fire before forward layer execution"""
            def hook(module, input):
                if name not in self.start_events:
                    self.start_events[name] = torch.cuda.Event(enable_timing=True)
                self.start_events[name].record()
            return hook
        
        def make_forward_post_hook(name):
            """Hook the fires after forward layer execution"""
            def hook(module, input, output):
                if name not in self.end_events:
                    self.end_events[name] = torch.cuda.Event(enable_timing=True)
                self.end_events[name].record()
            return hook
        
        for name, module in self.model.named_modules():
            pre_hook = module.register_forward_pre_hook(make_forward_pre_hook(name))
            post_hook = module.register_forward_hook(make_forward_post_hook(name))
            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

    def compute_timings(self):
        torch.cuda.synchronize()
        for name in self.start_events.keys():
            if name in self.end_events:
                self.layer_times[name].append(self.start_events[name].elapsed_time(self.end_events[name]))

    def run_benchmark(self, iterations, batch_size=1):
      
        input_tensor = torch.randn(batch_size, 3, 512, 512).cuda()
        self.warmup(self.model, input_tensor, iterations=20)
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for i in range(iterations):
            # torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            times_ms.append(elapsed_time)
            
            if (i+1)%50 == 0:
                print(f"{i+1} iterations complete")
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        if self.run_with_hooks:
            self.compute_timings()
            self.print_layer_profile()
        times_ms = np.array(times_ms)
        times_trimmed = times_ms[10:] #Trim times as first might be unstable
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
    
    def print_layer_profile(self):
        print(f"{'Layer Name':<50} {'Time (ms)':<12}")
        print("-"*80)
        
        total_layer_sum = 0.0
        for (name, metrics) in self.layer_times.items():
            print(f"{name:<50} {metrics[0]}")
            total_layer_sum += sum(metrics)
        
        print(f"Sum of inference time of all layers is {total_layer_sum}")
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __exit__(self):
        if self.hooks != []:
            self.remove_hooks()
    
def main():
    # opt = opts().parse()
    # opt.arch = 'dla_34'
    # opt.heads = {'hm': 80, 'wh': 2, 'reg': 2}
    # opt.head_conv = 256

    arch = "dla_34"
    heads = {'hm': 80, 'wh': 2, 'reg': 2}
    head_conv = 256 #64 for Pascal, 256 for COCO
    model = create_model(arch, heads, head_conv) #Head conv = 256
    model = load_model(model, '/home/orin/Dev/CenterNet/CenterNet-edge-testing/models/ctdet_pascal_dla_384.pth')
    model = model.cuda().eval()
    model = model.half()
    device = "cuda"
    num_iterations = 100
    run_with_hooks = True

    CenterNet_benchmark = Benchmark(model=model, device=device, run_with_hooks=run_with_hooks)
    '''Batch size based benchmarking'''
    # for batch_size in [1,2,4,8]:
    #     print(f"Batch size {batch_size}:\n")
    #     CenterNet_benchmark.run_benchmark(num_iterations, batch_size=batch_size)
    #     # print(f"Average = {avg:.2f}, Std = {std:.2f}\n")
    #     time.sleep(2)

    '''Layer Wise breakdwon'''
    CenterNet_benchmark.run_benchmark(num_iterations)

if __name__ == "__main__":
    main()