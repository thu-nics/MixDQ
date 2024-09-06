from collections import namedtuple
import torch
# import gemm_int8_tensorcore_test
from mixdq_extension.op.qlinear import qlinear
from mixdq_extension.op.qconv2d import qconv2d
from typing import Optional
import torch.nn.functional as F
import math

SCALE=0.1  # init the quant parameters


def quantize_per_tensor_uint4(
    input: torch.Tensor, scale, zero_point, 
):

    # reshape the quant parameters for quantizing 
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))
    
    # scale = scale.reshape()
    scale_inv = 1.0 / scale
    int_repr = torch.clamp(torch.round(input * scale_inv) + zero_point, 0, 15).to(torch.uint8)
    if len(input.shape) >= 4:
        assert input.shape[1] % 2 == 0
        return (int_repr[:, ::2, ...] << 4 | int_repr[:, 1::2, ...])
    assert input.shape[-1] % 2 == 0
    return (int_repr[...,::2] << 4 | int_repr[..., 1::2])


def unpack_uint4(input):
    shape = input.shape
    if len(shape) >= 4:
        packed_dim = 2
        new_shape = (input.shape[0], input.shape[1]*2, *input.shape[2:])
    else:
        packed_dim = -1
        new_shape = (*input.shape[:-1], input.shape[-1]*2)
    first_elements = (input >> 4).to(torch.uint8)
    second_elements = (input & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=packed_dim).view(new_shape)
    
    
def dequantize_per_tensor_uint4(
        input, scale, zero_point,
):
    # reshape the quant parameters for dequantizing 
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))
    
    input = unpack_uint4(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


dtype_to_bw = {
    torch.qint8: 8,
    torch.quint8:8,
    torch.quint4x2:4,
    torch.quint2x4:2,
    torch.float16:16,
}


class QParam(namedtuple("QParam", ["qscheme", "dtype", "scales", "zero_points", "axis"], defaults=[torch.per_tensor_affine, torch.quint8, 1.0, 0.0, 0])):
    @property
    def zp_float(self):
        return self.scales * self.zero_points
    pass
# class QParam:
#     def __init__(self, qscheme=torch.per_tensor_affine, dtype=torch.quint8, scales=1.0, zero_points=0.0, axis=0):
#         self.qscheme = qscheme
#         self.dtype = dtype
#         self.scales = scales
#         self.zero_points = zero_points
#         self.axis = axis
#         self.zp_float = scales*zero_points


def create_qparams_from_dtype(  
                            dtype, 
                            device, 
                            is_channel_wise=False, 
                            num_kernels=None,
                            ckpt = None, 
                            module_name=None, 
                            bit_width = 0,
                            quant_type=None, 
                            split=0,
):

    if dtype == torch.float16:
        return None
    elif dtype in [torch.qint8, torch.quint8, torch.quint4x2]:
        if quant_type == 'weight':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt, 
                                                                        bit_width, 
                                                                        module_name,
                                                                        quant_type='weight', 
                                                                        split=split,
                                                                        device=device)
        elif quant_type == 'act':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt, 
                                                                        bit_width, 
                                                                        module_name,
                                                                        quant_type='act', 
                                                                        split=split,
                                                                        device=device)
    else:
        raise ValueError(f"Unsupported quantize dtype {dtype}")

    if is_channel_wise:
        assert num_kernels is not None 
        qparam  = QParam(qscheme=torch.per_channel_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype, axis=0)
        if split > 0:
            qparam_0 = QParam(qscheme=torch.per_channel_affine,
                        scales=scales_0, zero_points=zero_points_0,
                        dtype=dtype, axis=0)
        else:
            qparam_0 = None
        
    else:
        qparam = QParam(qscheme=torch.per_tensor_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype)

        if split > 0:
            qparam_0 = QParam(qscheme=torch.per_tensor_affine,
                        scales=scales_0, zero_points=zero_points_0,
                        dtype=dtype)
        else:
            qparam_0 = None
        
    return qparam, qparam_0


def quantize_from_qparams(x: torch.Tensor, qparams: QParam):
    if qparams.dtype == torch.quint4x2:
        # TODO: support both per-channel and per-tensor
        # assert qparams.qscheme == torch.per_tensor_affine
        # print(x.shape)
        return quantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device))

    if qparams.qscheme in [torch.per_tensor_affine]:
        scales = qparams.scales
        scales = scales.clone().detach().to(x.device) \
                 if isinstance(scales, torch.Tensor) \
                 else torch.tensor(scales, dtype=torch.float16, device=x.device)
        zps = qparams.zero_points
        zps = zps.clone().detach().to(x.device) \
              if isinstance(zps, torch.Tensor) \
              else torch.tensor(zps, dtype=torch.float16, device=x.device)

        # Quantize only works on Float Tensor not Half. TODO: custom kernels
        x = x.to(torch.float32)
        x_quant = torch.quantize_per_tensor(x, scales, zps, qparams.dtype)
    elif qparams.qscheme in [torch.per_channel_affine]:
        scales = qparams.scales
        assert isinstance(scales, torch.Tensor)
        scales = scales.clone().detach().to(x.device)
        zps = qparams.zero_points
        assert isinstance(zps, torch.Tensor)
        zps = zps.clone().detach().to(x.device)
        assert qparams.axis < len(x.shape)
        # Quantize only works on Float Tensor not Half TODO: custom kernels
        x = x.to(torch.float32)
        # print(scales.shape)
        # if scales.shape == torch.Size([]):
        #     # torch.quantize_per_channel need the shape of scales and zps to be torch.size([N])
        #     scales = scales.reshape(1)
        #     zps = zps.reshape(1)
        x_quant = torch.quantize_per_channel(x, scales, zps, axis=qparams.axis,
                                             dtype=qparams.dtype)
    else:
        raise ValueError(f"Unknown qscheme {qparams.qscheme}")
    return x_quant


def dequantize_to_float16_linear(x: torch.Tensor, qparams: QParam):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float32)
    elif x.dtype in [torch.int8]:
        scale = (qparams.scales.view(-1, *([1] * (len(x.shape) - 1)))).cuda().float()
        zero_points = (qparams.zero_points.view(-1, *([1] * (len(x.shape) - 1)))).cuda().float()

        x = scale*(x- zero_points)
        return x

    assert x.dtype == torch.uint8 # the current way to support uint4
    return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device)).to(torch.float16)


def dequantize_to_float16(x: torch.Tensor, qparams: QParam):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float16)
    elif x.dtype in [torch.int8]:
        scale = (qparams.scales.view(-1, *([1] * (len(x.shape) - 1)))).cuda()
        zero_points = (qparams.zero_points.view(-1, *([1] * (len(x.shape) - 1)))).cuda()

        x = scale*(x- zero_points)
        return x

    assert x.dtype == torch.uint8 # the current way to support uint4
    return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device)).to(torch.float16)


def linear_on_quantized_data(
        w_tensor: torch.Tensor = None,
        w_tensor_org: torch.Tensor = None,
        w_qparams: QParam = None,
        key_first_token: torch.Tensor = None,
        a_tensor: torch.Tensor = None,
        a_qparams: QParam = None,
        bias: Optional[torch.Tensor] = None,
        bos: bool = False,
        module_name = None,
        bos_pre_computed = None,
        # k_tensor_text = None,
        # v_tensor_text = None
) -> torch.Tensor:
    if not bos:
        # functional simulation for now (TODO: kernel support)
        if a_qparams is not None:
            out = gemm_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias)
            return out  # , _
            
        else:
            # out, _ = gemm_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias)
            # a_tensor_org = a_tensor
            # w_tensor_org = w_tensor
            # bias_org = bias
            # a_tensor = dequantize_to_float16_linear(a_tensor, a_qparams) if a_qparams is not None else a_tensor.float()
            # w_tensor = dequantize_to_float16_linear(w_tensor, w_qparams)
            # bias = bias.float() if bias is not None else bias
            # output = F.linear(a_tensor, w_tensor, bias).half()
            # torch.testing.assert_close(output, _)
            # return output # F.linear(a_tensor, w_tensor, bias).half()
            a_tensor = dequantize_to_float16(a_tensor, a_qparams) if a_qparams is not None else a_tensor
            w_tensor = dequantize_to_float16(w_tensor, w_qparams)
            return F.linear(a_tensor, w_tensor, bias)
            
    else:
        print("apply bos!")
        
        # TODO: pre-compute the first token or not
        # compute the first token and the the others seperately
        # out_0 = F.linear(key_first_token.unsqueeze(1), w_tensor_org, bias)
        # TODO:Note that batch_size of the bos_pre_computed is 1, if bs!=1, out_0 should be repeated
        out_0 = bos_pre_computed.cuda()
        # a_tensor = dequantize_to_float16_linear(a_tensor, a_qparams)
        # w_tensor = dequantize_to_float16_linear(w_tensor, w_qparams)
        # bias = bias.float() if bias is not None else bias
        # out_1 = F.linear(a_tensor, w_tensor, bias).half()
        out_1 = gemm_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias)
        out_0 = out_0.expand(out_1.shape[0], -1, -1)
        return torch.cat([out_0, out_1],dim=1)  # , torch.cat([out_0, _],dim=1)


def conv2d_on_quantized_data(
        w_tensor: torch.Tensor = None,
        w_tensor_0: torch.Tensor = None,
        w_qparams: QParam = None,
        w_qparams_0: QParam = None,
        a_tensor: torch.Tensor = None,
        a_tensor_0: torch.Tensor = None,
        a_qparams: QParam = None,
        a_qparams_0: QParam = None,
        bias: Optional[torch.Tensor] = None,
        stride=1,
        padding=0, 
        dilation=1,
        groups=1,
        split=0
) -> torch.Tensor:
    # functional simulation for now (TODO: kernel support)
    if split==0:
        if a_qparams is not None:
            out = conv_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias, stride, padding, dilation, groups)
            return out

        else:
            a_tensor = dequantize_to_float16(a_tensor, a_qparams) if a_qparams is not None else a_tensor
            w_tensor = dequantize_to_float16(w_tensor, w_qparams)
            return F.conv2d(a_tensor, w_tensor, bias, stride, padding, dilation, groups)

    elif split > 0:
        if a_qparams is not None:
            # weight = dequantize_to_float16(w_tensor, w_qparams)
            # weight_0 = dequantize_to_float16(w_tensor_0, w_qparams_0)
            # input = dequantize_to_float16(a_tensor, a_qparams)
            # input_0 = dequantize_to_float16(a_tensor_0, a_qparams_0)
            # a_tensor = torch.cat([input, input_0], dim=1)  if a_qparams_0 is not None else a_tensor
            # out = F.conv2d(input, weight, None, stride, padding, dilation, groups)
            # out_0 = F.conv2d(input_0, weight_0, None, stride, padding, dilation, groups)
            out = conv_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, None, stride, padding, dilation, groups)
            out_0 = conv_cutlass(w_qparams_0, a_qparams_0, w_tensor_0, a_tensor_0, None, stride, padding, dilation, groups)

            shape = bias.size()
            bias = bias.reshape(1,shape[0],1,1)
            out = out + out_0 + bias

        else:
            weight = dequantize_to_float16(w_tensor, w_qparams)
            weight_0 = dequantize_to_float16(w_tensor_0, w_qparams_0)
            a_tensor = a_tensor
            w_tensor = torch.cat([weight, weight_0], dim=1)
            out = F.conv2d(a_tensor, w_tensor, bias, stride, padding, dilation, groups)

        # w_tensor = torch.cat([weight, weight_0], dim=1)
        return out


def gemm_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias):
    s_w = w_qparams.scales.cuda().float()
    s_a = a_qparams.scales.cuda().float()
    z_a = a_qparams.zero_points.cuda().float()
    zps_a = a_qparams.zp_float.cuda().float()

    a_int = a_tensor
    w_int = w_tensor.int_repr()  # if w_tensor.dtype is torch.qint8 else w_tensor.transpose(0,1)

    output_ref = qlinear(
        a_int, 
        w_int,
        s_w,
        s_a,
        z_a,
        bias
    )

    # original_size = a_int.size()

    # if len(original_size)>2:
    #     # reshape
    #     a_int = a_int.view(-1, original_size[-1])

    # # reshape the matrix
    # _, s_w = torch.broadcast_tensors(w_int, s_w)
    # _, s_a = torch.broadcast_tensors(a_int, s_a)
    # _, zps_a = torch.broadcast_tensors(a_int, zps_a)

    # # output = gemm_int8_tensorcore_test.run(a_tensor, w_tensor)  the shape of the tensor should be [xx, in_features]
    # out_int = a_int.to(torch.float32)@w_int.to(torch.float32)
    # inf_check = torch.isinf(out_int)
    # has_inf = torch.any(inf_check)
    # assert not has_inf, "there are inf in the tensor!"

    # output = (s_a@s_w)/s_w.shape[0]*out_int
    # inf_check = torch.isinf(output)
    # has_inf = torch.any(inf_check)
    # assert not has_inf, "there are inf in the tensor!"

    # output = output - zps_a@(s_w*w_int)  # a_int = (a_float+zps_a)/s   a_int:[-128,127]

    # if bias is not None:
    #     output = output+bias
    
    # if len(original_size)>2:
    #     output = output.view(*original_size[:-1], w_int.size(1))
    
    # output = output.to(torch.float16)

    # inf_check = torch.isinf(output)
    # has_inf = torch.any(inf_check)
    # assert not has_inf, "there are inf in the tensor!"
    print("run gemm on tensor core")

    # torch.testing.assert_close(output, output_ref)
    return output_ref
    

def conv_cutlass(w_qparams, a_qparams, w_tensor, a_tensor, bias, stride, padding, dilation, groups):
    print("run qconv2d!")
    s_w = w_qparams.scales.cuda().to(torch.float32)
    s_a = a_qparams.scales.cuda().to(torch.float32)
    z_a = a_qparams.zero_points.cuda().to(torch.float32)
    zps_a = a_qparams.zp_float.cuda().to(torch.float32)

    a_int = a_tensor
    w_int = w_tensor.int_repr()
    
    a_int = a_int.to(memory_format=torch.channels_last)
    w_int = w_int.to(memory_format=torch.channels_last)

    if len(set(padding)) == 1:
        padding = padding[0]
    else:
        raise RuntimeError("the padding has different elements")
    if len(set(stride)) == 1:
        stride = stride[0]
    else:
        raise RuntimeError("the stride has different elements")

    output = qconv2d(
        a_int, 
        w_int,
        s_w,
        s_a,
        z_a,
        bias,
        stride,
        padding,)

    return output


def get_quant_para(ckpt, n_bit, module_name, quant_type, split=0, device=None):

    if split == 0:
        bit_idx = int(math.log2(n_bit)-1)

        if quant_type == 'weight':
            module_name = module_name + '.weight_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]  # sym quantization, zp=0
            # print(zero_point)

        elif quant_type == 'act':
            module_name = module_name + '.act_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx] -128  # change the data type from uint8 to int8

        return scales.to(device), zero_point.to(device), None, None

    elif split > 0:
        bit_idx = int(math.log2(n_bit)-1)

        if quant_type == 'weight':
            module_name = module_name + '.weight_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]

            module_name = module_name + '_0'
            assert module_name in ckpt.keys()
            scales_0 = ckpt[module_name]['delta_list'][bit_idx]
            zero_point_0 = ckpt[module_name]['zero_point_list'][bit_idx]
            # print(zero_point, zero_point_0)

        elif quant_type == 'act':
            module_name = module_name + '.act_quantizer'

            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]-128

            module_name = module_name + '_0'
            assert module_name in ckpt.keys()
            scales_0 = ckpt[module_name]['delta_list'][bit_idx]
            zero_point_0 = ckpt[module_name]['zero_point_list'][bit_idx]-128

        return scales.to(device), zero_point.to(device), scales_0.to(device), zero_point_0.to(device)