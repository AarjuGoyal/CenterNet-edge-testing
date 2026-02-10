import torch
import torch.nn as nn
import numpy as np
import time
import copy
from lib.models.model import create_model, load_model

def fuse_conv_bn_debug(conv, bn):
   
    # Get conv weights and bias
    W = conv.weight.data.clone()
    if conv.bias is not None:
        b = conv.bias.data.clone()
    else:
        b = torch.zeros(conv.out_channels, device=conv.weight.device)
    
    # Get BN parameters
    gamma = bn.weight.data.clone()
    beta = bn.bias.data.clone()
    mean = bn.running_mean.clone()
    var = bn.running_var.clone()
    eps = bn.eps
    
    # Compute fused parameters
    # Scale factor: gamma / sqrt(var + eps)
    scale_factor = gamma / torch.sqrt(var + eps)
    print(f"\nScale factor (first 3): {scale_factor[:3]}")
    
    # New weights: W * scale_factor (broadcast over output channels)
    W_fused = W * scale_factor.view(-1, 1, 1, 1)
    print(f"Fused weight shape: {W_fused.shape}")
    print(f"Fused weight magnitude change: {W.abs().mean():.6f} → {W_fused.abs().mean():.6f}")

    # New bias: (b_conv - mean) * scale_factor + beta
    print(f"Shape of conv_bias {b.shape}, shape of gamma {gamma.shape}, mean {mean.shape}, W_fused {W_fused.shape}")
    b_fused = (b* gamma) - (mean * W_fused) + beta
    print(f"\nFused bias (first 3): {b_fused[:3]}")
    print(f"Original bias (first 3): {b[:3]}")

    # Create new conv layer with fused parameters
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # Always have bias after fusion
        padding_mode=conv.padding_mode
    )
    
    # Load fused parameters
    fused_conv.weight.data = W_fused
    fused_conv.bias.data = b_fused
    
    return fused_conv

def fuse_conv_bn(conv, bn):
    """
    Fuse a Conv2d and BatchNorm2d layer.
    
    Math:
        y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
        
    Can be rewritten as:
        y = conv_new(x)
        where:
            W_new = W_conv * gamma / sqrt(var + eps)
            b_new = (b_conv - mean) * gamma / sqrt(var + eps) + beta
    
    Args:
        conv: nn.Conv2d layer
        bn: nn.BatchNorm2d layer
    
    Returns:
        Fused conv layer with BN parameters absorbed
    """
    # Get conv weights and bias
    conv_weight = conv.weight.clone()
    if conv.bias is not None:
        conv_bias = conv.bias.clone()
    else:
        conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device)
    
    # Get BN parameters
    bn_weight = bn.weight.clone() #Gamma
    bn_bias = bn.bias.clone() #Beta
    bn_mean = bn.running_mean.clone()
    bn_var = bn.running_var.clone()
    bn_eps = bn.eps
    
    # Compute fused parameters
    # Scale factor: gamma / sqrt(var + eps)
    scale_factor = bn_weight / torch.sqrt(bn_var + bn_eps)
    
    # New weights: W * scale_factor (broadcast over output channels)
    fused_weight = conv_weight * scale_factor.view(-1, 1, 1, 1)
    
    # New bias: (b_conv - mean) * scale_factor + beta
   
    fused_bias = (conv_bias* bn_weight) - (bn_mean * fused_weight) + bn_bias
    
    # Create new conv layer with fused parameters
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # Always have bias after fusion
        padding_mode=conv.padding_mode
    )
    
    # Load fused parameters
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    
    return fused_conv

CONV_BN_PAIRS = [
    ('conv1', 'bn1'),
    ('conv2', 'bn2'),
    ('conv3', 'bn3'),
    ('conv',  'bn'),          # DLA root layers use this
]

def fuse_module_inplace(module):
    """
    Scan a single module's direct attributes for Conv-BN pairs
    and replace them in-place.
    
    Returns the number of pairs fused at this level.
    """
    fused = 0
    for conv_name, bn_name in CONV_BN_PAIRS:
        conv = getattr(module, conv_name, None)
        bn   = getattr(module, bn_name,  None)

        # Both must exist AND be the right types
        if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
            continue

        # Fuse and overwrite the conv attribute; remove the bn attribute
        fused_conv = fuse_conv_bn(conv, bn)
        setattr(module, conv_name, fused_conv)
        setattr(module, bn_name,   nn.Identity())   # replace BN with no-op
        fused += 1
        print(f"    ✓ Fused {conv_name}/{bn_name} in {module.__class__.__name__}")

    return fused

def fuse_model_bn(model):
    """
    Recursively walk every module in the model and fuse Conv-BN pairs.
    
    Works with ANY container type:
      - nn.Sequential
      - Custom classes (DLATree, DLARoot, IDA, etc.)
      - Nested combinations of both
    """
    model = copy.deepcopy(model)
    model.eval()   # BN running stats must be in eval mode

    total_fused = 0

    # model.modules() gives EVERY module in the tree (depth-first)
    # We check each one for conv/bn attribute pairs
    for name, module in model.named_modules():
        n = fuse_module_inplace(module)
        total_fused += n

    print(f"\n  Total: fused {total_fused} Conv-BN pairs")
    return model


def print_modules(model):

    for module in model.modules():
        print("-------------------------------")
        print(module)
def count_bn_layers(model):
    """Count BatchNorm layers in model."""
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

def verify_numerical_equivalence(original, fused, input_tensor, tol=1e-3):
    """Compare outputs of original vs fused model."""
    original.eval()
    fused.eval()

    with torch.no_grad():
        out_orig  = original(input_tensor)
        out_fused = fused(input_tensor)

    # CenterNet returns a list of dicts
    if isinstance(out_orig, list):
        out_orig  = out_orig[0]
        out_fused = out_fused[0]

    print("\n  Output comparison (per head):")
    all_ok = True
    for key in out_orig:
        max_diff  = (out_orig[key] - out_fused[key]).abs().max().item()
        mean_diff = (out_orig[key] - out_fused[key]).abs().mean().item()
        ok = max_diff < tol
        status = "✓" if ok else "⚠️"
        print(f"    {status} {key}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        if not ok:
            all_ok = False

    return all_ok

def main():
    arch = "dla_34"
    heads = {'hm': 80, 'wh': 2, 'reg': 2}
    head_conv = 256 #64 for Pascal, 256 for COCO
    model = create_model(arch, heads, head_conv) #Head conv = 256
    model = load_model(model, '/home/orin/Dev/CenterNet/CenterNet-edge-testing/models/ctdet_pascal_dla_384.pth')

    model = model.cuda().eval()
    count_bn = count_bn_layers(model=model)
    
    input_tensor = torch.randn(1, 3, 512, 512).cuda()
    print(f"Before Fusing:\n ")
    print(f"Number of batch norm layers: {count_bn}")
    # print_modules(model=model)
    
    # print("Number of batch norm layers", count_bn)
    
    # ##FUSE MODEL
    # fused_bn_model = fuse_model_bn(model=model)

    # #After Fusing
    # print(f"After Fusing:\n ")
    # count_bn_af = count_bn_layers(fused_bn_model)
    

    # # print_modules(fused_bn_model)
    # print(f"Number of batch norm layers: {count_bn_af}")

    print("Sanity check if the correct layers are fused and generate the same output")

    original_module = model.base.level3.tree1.tree2
    conv = original_module.conv1
    bn = original_module.bn1
    conv.eval()
    bn.eval()
    fusion_test_input = torch.randn(1, conv.in_channels, 32 , 32).cuda()

    print(f"before fusion\n")
    print(f"conv1: {type(original_module.conv1)}")
    print(f"bn1: {type(original_module.bn1)}")
    with torch.no_grad():
        original_out = bn(conv(fusion_test_input))

    print("Conv in eval mode?", not conv.training)
    print("BN in eval mode?", not bn.training)

    fused_conv = fuse_conv_bn_debug(conv, bn)
    fused_conv.eval()
    # fused_module = fused_bn_model.base.level2.tree1
    # fused_conv= fused_module.conv1
    # fused_conv.eval()
    print(f"after fusion")
    # print(f" conv1: {type(fused_module.conv1)}")
    # print(f" bn1: {type(fused_module.bn1)}")

    with torch.no_grad():
        fused_out = fused_conv(fusion_test_input)

    diff = (original_out - fused_out).abs().max()
    print(f"MAx difference: {diff:.2e}")
    
    # print(verify_numerical_equivalence(original=model, fused=fused_bn_model, input_tensor=input_tensor, tol=1e-4))
if __name__ == "__main__":
    main()
