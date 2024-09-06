import torch
import mixdq_extension._C
from .quant import quantize_per_tensor, quantize_per_tensor_vectorized

qlinear = mixdq_extension._C.qlinear_w8_a8_ohalf
qlinear_ref =  mixdq_extension._C.qlinear_fp_reference

# def qlinear(input_int,
#             weight_int,
#             weight_scale,
#             input_scale,
#             input_zero_point,
#             unused0,
#             unused1,
#             unused2,
#             bias=None):
#     weight_fp = weight_int.float() * weight_scale[:, None]
#     input_fp = (input_int.float() - input_zero_point) * input_scale
#     return torch.nn.functional.linear(
#         input_fp.half(),
#         weight_fp.half(),
#         bias
#     )

# from pytorch_lightning import seed_everything


if __name__ == "__main__":
    # seed_everything(42)

    def run_test(nsamples, ic, oc, use_bias=True):

        dev='cuda'
        input_fp16 = 6 * torch.rand(nsamples, ic, dtype=torch.float16, device=dev) - 3

        # quantized = quantize_per_tensor(t, scale, zp)
        # input_fp16 = torch.randint(-3, 3, (nsamples, ic), dtype=torch.int8, device=dev).to(torch.float16)
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
        input_int_2 = quantize_per_tensor_vectorized(input_fp16, input_scale, input_zp).to(torch.int8)
        torch.testing.assert_close(input_int, input_int_2, atol=1., rtol=1e-2)

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

        def get_reference_int_compute():
            infused_scale = weight_scale * input_scale
            offset = weight_scale * weight_int.to(torch.int32).sum(dim=1)
            offset *= input_zp * input_scale
            infused_bias = bias.to(torch.float32) - offset
            int_gemm_out = torch.matmul(input_int.to(torch.float32), 
                            weight_int.to(torch.float32).transpose(0, 1))
            output = int_gemm_out * infused_scale - offset + bias.float()
            return output.to(torch.float16)

        def get_reference_fp_compute_torch():
            weight_fp = weight_int.to(torch.float32) * weight_scale[:, None]
            input_fp = (input_int.to(torch.float32) - input_zp) * input_scale
            # debug_only: with all ones.
            fp_output = torch.matmul(input_fp, weight_fp.transpose(0, 1))
            output = fp_output + bias.float()
            output = output.half()
            return output
        
        def test_reference_fp_compute_cutlass():
            print(f"nsamples: {nsamples}, input_channel: {ic}, output_channel: {oc}")

            # debug_only: 
            weight_fp = 0.158**torch.rand((ic,oc), device=dev, dtype=torch.float16)
            input_fp = 0.158*torch.rand((nsamples,ic), device=dev, dtype=torch.float16)

            fp_output = qlinear_ref(input_fp, weight_fp, bias) # bias no use, but have to feed in; weight_fp columnMajor like int
            fp_output_ = torch.matmul(input_fp, weight_fp)
            print(fp_output,'\n', fp_output_)
            torch.testing.assert_close(fp_output, fp_output_, rtol=1e-4, atol=1e-2)
        
        reference_int = get_reference_int_compute()
        reference_fp_torch = get_reference_fp_compute_torch()

        torch.testing.assert_close(output, reference_int, atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(output, reference_fp_torch, rtol=1e-2, atol=1e-2)

        reference_fp_cutlass = test_reference_fp_compute_cutlass()
        

    # run_test(1024, 16, 256)
    # run_test(64, 8, 16)    # the cutlass_fp16_refence kernel have alignment of 8, could not support 4
    run_test(64, 8, 16)
