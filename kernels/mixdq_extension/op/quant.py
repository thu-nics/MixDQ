import torch
import mixdq_extension._C

quantize_per_tensor = mixdq_extension._C.quantize_per_tensor_to_int8
quantize_per_tensor_vectorized = mixdq_extension._C.quantize_per_tensor_to_int8_vectorized

if __name__ == "__main__":
    size = (1024, )
    t = torch.rand(size, dtype=torch.float16, device='cuda')

    def get_quant_params(tensor):
        """An example algorithm to obtain zp and scale. """
        tensor = tensor.to(torch.float32)
        zero_point = (torch.max(tensor) + torch.min(tensor))/2
        zero_point = torch.round(zero_point)
        scale = (torch.max(tensor) - torch.min(tensor)) / 255
        return scale, zero_point
    
    scale, zp = get_quant_params(t)

    quantized = quantize_per_tensor(t, 1/scale, zp)
    quantized_2 = quantize_per_tensor_vectorized(t, 1/scale, zp)

    reference = torch.quantize_per_tensor(input=t.float(), 
                                          scale=scale, 
                                          zero_point=zp, 
                                          dtype=torch.qint8).int_repr()

    torch.testing.assert_close(quantized, reference)
    torch.testing.assert_close(quantized, quantized_2)

    def test_cuda_graph():
        static_input = (t.detach().clone(), 1/scale.detach().clone(), zp.detach().clone())
        static_output = torch.empty_like(quantized)
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_output = quantize_per_tensor(*static_input)
        torch.cuda.current_stream().wait_stream(s)

        # import pdb; pdb.set_trace()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = quantize_per_tensor(*static_input)

        # replay
        tt = torch.rand(size, dtype=torch.float16, device='cuda')
        
        scale, zp = get_quant_params(tt)

        static_input[0].copy_(tt)
        static_input[1].copy_(1/scale)
        static_input[2].copy_(zp)
        g.replay()

        reference = torch.quantize_per_tensor(tt.float(), scale, zp, dtype=torch.qint8).int_repr()
        torch.testing.assert_close(static_output, reference)
    
    # test_cuda_graph()