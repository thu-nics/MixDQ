import torch
import mixdq_extension._C

def qconv2d(
    input_int,
    weight_int,
    weight_scale,
    input_scale,
    input_zp,
    scale,
    weight_sum_by_input_channels,
    bias0,
    bias=None,
    stride=1,
    padding=0,
):
    dilation = 1
    return mixdq_extension._C.qconv2d_w8_a8_ohalf(
        input_int, weight_int, weight_scale, input_scale, input_zp,
        scale, weight_sum_by_input_channels, bias0, 
        bias, stride, padding, dilation
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    def run_test(n, h, w, c, k, r, s, pad, stride, dilation=1, bias=True):
        dev = torch.device('cuda:0')

        input_int = torch.randint(-3, 3, 
                                  (n, c, h, w), 
                                  dtype=torch.int8, 
                                  device=dev)
        input_int = input_int.to(memory_format=torch.channels_last)

        weight_int = torch.randint(-3, 3, 
                                   (k, c, r, s),
                                   dtype=torch.int8, 
                                   device=dev)
        weight_int = weight_int.to(memory_format=torch.channels_last)

        weight_scale = 0.1 + torch.rand((k,), dtype=torch.float32, device=dev)
        input_scale = torch.tensor(0.123, dtype=torch.float32, device=dev)
        input_zp = torch.tensor(2.345, dtype=torch.float32, device=dev)

        if bias:
            bias_ = torch.rand((k,), device=dev, dtype=torch.float16)
        else:
            bias_ = None

        output = qconv2d(
            input_int, 
            weight_int,
            weight_scale,
            input_scale,
            input_zp,
            weight_scale*input_scale,
            weight_int.float().sum(dim=1, keepdim=True) if pad>0 else None,
            weight_int.float().sum(dim=[1,2,3])*input_zp if pad==0 else None,
            bias_,
            stride,
            pad,)
        
        def get_reference_int_compute():
            accumulator = F.conv2d(
                input_int.to(torch.float32),
                weight_int.to(torch.float32),
                stride=stride,
                padding=pad,
                dilation=dilation
            )
            
            # bias1
            w_ = weight_int.to(torch.float32).sum(dim=1, keepdim=True)
            a_ = torch.broadcast_to(-1*input_zp, (input_int.shape[0], 1, input_int.shape[2], input_int.shape[3]))
            bias0 = F.conv2d(a_, w_, stride=stride, padding=pad, dilation=dilation)
            
            output = (accumulator + bias0)*weight_scale[None, :, None, None] * input_scale
            
            if bias:
                output += bias_[None, :, None, None]
            return output.to(torch.float16)
        
        def get_reference_fp_compute():
            weight_fp = weight_int.to(torch.float32) * weight_scale[:, None, None, None]
            input_fp = (input_int.to(torch.float32) - input_zp) * input_scale
            fp_output = F.conv2d(input_fp, weight_fp,
                stride=stride,
                padding=pad,
                dilation=dilation)
            output = fp_output.to(torch.float16)
            if bias:
                output += bias_[None, :, None, None]
            return output

        reference_int = get_reference_int_compute()
        reference_fp = get_reference_fp_compute()

        torch.testing.assert_close(reference_int, reference_fp, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(output, reference_int)
    

    run_test(1, 14, 14, 512, 1024, 3, 3, pad=1, stride=1, bias=True)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=1, stride=1, bias=True)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=1, stride=2, bias=True)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=0, stride=1, bias=False)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=0, stride=1, bias=True)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=0, stride=2, bias=True)
    run_test(1, 14, 14, 512, 1024, 3, 3, pad=0, stride=1, bias=False)
    # small alignment test cases
    run_test(1, 7, 7, 4, 320, 3, 3, pad=1, stride=1, bias=True) 
    run_test(1, 7, 7, 4, 320, 3, 3, pad=0, stride=1, bias=True) 
    run_test(1, 7, 7, 4, 320, 3, 3, pad=1, stride=2, bias=True) 
    run_test(1, 7, 7, 4, 320, 3, 3, pad=0, stride=2, bias=True) 
    run_test(1, 7, 7, 320, 4, 3, 3, pad=1, stride=1, bias=True)
    run_test(1, 7, 7, 320, 4, 3, 3, pad=0, stride=1, bias=True)
    run_test(1, 7, 7, 320, 4, 3, 3, pad=1, stride=2, bias=True)
    run_test(1, 7, 7, 320, 4, 3, 3, pad=0, stride=2, bias=True)
    ### dilation has bugs. DOn't use dilation > 1
    # run_test(1, 14, 14, 512, 1024, 3, 3, pad=0, stride=1, dilation=2, bias=False)
    # run_test(1, 14, 14, 512, 1024, 3, 3, pad=1, stride=2, dilation=2, bias=False)
    # run_test(1, 14, 14, 512, 1024, 3, 3, pad=1, stride=2, dilation=2, bias=True)

    