import torch
import mixdq_extension._C
from mixdq_extension.op.quant import quantize_per_tensor

nsamples = 1280
ic = 1024
oc = 512
use_bias = True

qlinear = mixdq_extension._C.qlinear_w8_a8_ohalf
qlinear_ref =  mixdq_extension._C.qlinear_fp_reference

dev='cuda'
# input_fp16 = 6 * torch.rand(nsamples, ic, dtype=torch.float16, device=dev) - 3

# quantized = quantize_per_tensor(t, scale, zp)
input_fp16 = torch.randint(-3, 3, (nsamples, ic), dtype=torch.int8, 
                        device=dev).to(torch.float16)
weight_int = torch.randint(-3, 3, (oc, ic), dtype=torch.int8, device=dev)
input_scale = torch.scalar_tensor(0.123, dtype=torch.float32, device=dev)
input_zp = torch.scalar_tensor(5.00, dtype=torch.float32, device=dev)
# input_scale = torch.scalar_tensor(1, dtype=torch.float32, device=dev)
# input_zp = torch.scalar_tensor(0, dtype=torch.float32, device=dev)

weight_scale = 0.1 + torch.rand((oc,), dtype=torch.float32, device=dev)
if use_bias :
    bias = torch.rand((oc,), device=dev, dtype=torch.float16)
else:
    bias = None

input_int = quantize_per_tensor(input_fp16, input_scale, input_zp).to(torch.int8)

torch.cuda.cudart().cudaProfilerStart()
for i in range(5):
    torch.cuda.nvtx.range_push(f"iter_{i}")

    name_ = 'ours_int8'
    torch.cuda.nvtx.range_push(name_)
    with torch.autograd.profiler.emit_nvtx():
        output = qlinear(
        input_int, 
        weight_int,
        weight_scale,
        input_scale,
        input_zp,
        weight_int.float().sum(dim=1),
        weight_scale*input_scale,
        weight_int.float().sum(dim=1)*input_zp,
        bias
    )
    torch.cuda.nvtx.range_pop()

    name_ = 'cutlass_fp16'
    torch.cuda.nvtx.range_push(name_)
    fp_output = qlinear_ref(input_fp16, (weight_int).to(torch.float16), bias)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
