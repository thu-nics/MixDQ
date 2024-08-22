import torch.nn as nn
import torch
from torch.ao.quantization import QConfig
import torch.nn.functional as F
from .utils import (create_qparams_from_dtype, dtype_to_bw)
from mixdq_extension.op.quant import quantize_per_tensor, quantize_per_tensor_vectorized
from mixdq_extension.op.qconv2d import qconv2d
import logging

all = [
    'QuantizedConv2d'
]

quant_op = quantize_per_tensor_vectorized

class QuantizedConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, 
                 stride, padding, dilation, groups=1, bias=True,
                 device=None,
                 w_qparams=None, w_qparams_0=None, a_qparams=None, 
                 a_qparams_0 = None, module_name=None, split=0) -> None:
        super().__init__()

        self.module_name = module_name
        self.split = split  # for shortcut layer

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
        self.valid_for_acceleration = (
            w_qparams is not None and \
            a_qparams is not None and \
            w_qparams.dtype in [torch.qint8, torch.quint8] and \
            a_qparams.dtype in [torch.qint8, torch.quint8] and \
            w_qparams.qscheme == torch.per_channel_affine and \
            a_qparams.qscheme == torch.per_tensor_affine and \
            torch.all(w_qparams.zero_points == 0.0).item() and \
            (
                split == 0 or (
                    w_qparams_0 is not None and \
                    a_qparams_0 is not None and \
                    w_qparams_0.dtype in [torch.qint8, torch.quint8] and \
                    a_qparams_0.dtype in [torch.qint8, torch.quint8] and \
                    w_qparams_0.qscheme == torch.per_channel_affine and \
                    a_qparams_0.qscheme == torch.per_tensor_affine and \
                    torch.all(w_qparams_0.zero_points == 0.0).item()
                )
            ) and \
            (
                len(set(self.stride)) == 1 and len(set(self.padding)) == 1 and \
                len(set(self.dilation)) == 1 and self.dilation[0] == 1 and \
                self.groups == 1
            )
        )
        if self.valid_for_acceleration and (
            self.in_channels % 4 != 0 or self.out_channels % 4 != 0):
            logging.warning("Linear layer with in_features = "
                    f"{self.in_channels} and out_features = "
                    f"{self.out_channels} cannot use quantized kernel due to "
                    "misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False

        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", 
                                 w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", 
                                 w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", 
                                 a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", 
                                 a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
            if self.split != 0:
                self.register_buffer("weight_scales_0", 
                                    w_qparams_0.scales.to(device).float())
                self.register_buffer("weight_zero_points_0", 
                                    w_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_0", 
                                    a_qparams_0.scales.to(device).float())
                self.register_buffer("act_zero_points_0", 
                                    a_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_inv_0", 1 / self.act_scales_0)

    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, 
                                                            QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device=float_mod.weight.device
        # init the w & a quant parameters
        # split = 0
        # if split == 0:
            # init the quant parameters
        w_qparams, w_qparams_0 = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels,
                                                ckpt=ckpt,
                                                module_name=float_mod.module_name,
                                                quant_type='weight',
                                                bit_width=float_mod.w_bit,
                                                split=split)


        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype
        
        # if split == 0:
        if hasattr(float_mod, 'a_bit'):
            # if we want to quantized the act
            a_qparams, a_qparams_0 = create_qparams_from_dtype(dtype=act_dtype, 
                                                    device=device,
                                                    is_channel_wise=False,
                                                    num_kernels=num_kernels,
                                                    ckpt=ckpt,
                                                    module_name=float_mod.module_name,
                                                    quant_type='act',
                                                    bit_width=float_mod.a_bit,
                                                    split=split)
        else:
            a_qparams = None
            a_qparams_0 = None
            
        new_mod = cls(float_mod.in_channels,
                      float_mod.out_channels,
                      float_mod.kernel_size,
                      float_mod.stride,
                      float_mod.padding,
                      float_mod.dilation,
                      float_mod.groups,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,

                      w_qparams=w_qparams,
                      w_qparams_0 = w_qparams_0,
                      a_qparams=a_qparams,
                      a_qparams_0 = a_qparams_0,

                      module_name=float_mod.module_name,
                      split = split
                      )

        weight = float_mod.weight.detach()

        if split == 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(
                    weight.float(), 
                    new_mod.weight_scales, 
                    new_mod.weight_zero_points,
                    axis=w_qparams.axis, 
                    dtype=w_qparams.dtype).int_repr()

                new_mod.register_buffer("weight_int", weight_int)
                # auxiliary structure, used to quickly compute act_zp @ weight
                if float_mod.padding[0] == 0:
                    weight_sum_per_output_channel = \
                        weight_int.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0", 
                        weight_sum_per_output_channel*new_mod.act_zero_points)
                    new_mod.weight_sum_by_input_channels = None
                else:
                    weight_sum_by_input_channels = \
                        weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", 
                                            weight_sum_by_input_channels)
                    new_mod.bias0 = None
                new_mod.register_buffer("scale", 
                    new_mod.weight_scales * new_mod.act_scales)
            else:
                new_mod.register_buffer("weight", weight)            
            if float_mod.bias is not None:
                bias = float_mod.bias.detach()
                new_mod.register_buffer("bias", bias)
            else:
                new_mod.bias = None

        # for the weight of the shortcut
        elif split > 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(
                    weight[:, :split, ...].float(), 
                    new_mod.weight_scales, 
                    new_mod.weight_zero_points,
                    axis=w_qparams.axis, 
                    dtype=w_qparams.dtype).int_repr()

                weight_int_0 = torch.quantize_per_channel(
                    weight[:, split:, ...].float(), 
                    new_mod.weight_scales_0, 
                    new_mod.weight_zero_points_0,
                    axis=w_qparams_0.axis, 
                    dtype=w_qparams_0.dtype).int_repr()

                new_mod.register_buffer("weight_int", weight_int)
                new_mod.register_buffer("weight_int_0", weight_int_0)
                
                # auxiliary structure, used to quickly compute act_zp @ weight
                if float_mod.padding[0] == 0:
                    weight_sum_per_output_channel = \
                        weight_int.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0", 
                        weight_sum_per_output_channel * new_mod.act_zero_points)
                    weight_sum_per_output_channel_0 = \
                        weight_int_0.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0_0", 
                        weight_sum_per_output_channel_0 * new_mod.act_zero_points_0)
                    new_mod.weight_sum_by_input_channels = None
                    new_mod.weight_sum_by_input_channels_0 = None
                else:
                    weight_sum_by_input_channels = \
                        weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", 
                                            weight_sum_by_input_channels)
                    weight_sum_by_input_channels_0 = \
                        weight_int_0.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels_0", 
                                            weight_sum_by_input_channels_0)
                    new_mod.bias0 = None
                    new_mod.bias0_0 = None
                new_mod.register_buffer("scale", 
                    new_mod.weight_scales * new_mod.act_scales)
                new_mod.register_buffer("scale_0", 
                    new_mod.weight_scales_0 * new_mod.act_scales_0)
            else:
                new_mod.register_buffer("weight", weight)

            if float_mod.bias is not None:
                bias = float_mod.bias.detach()
                new_mod.register_buffer("bias", bias)
            else:
                new_mod.bias = None

        return new_mod
    
    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedConv2dW8A8"
        return "QuantizedConv2dFPFallback"
    
    def forward_fallback(self, x: torch.Tensor):
        weight_recovered = \
            self.weight_int.float() * self.weight_scales[:, None, None, None]
        weight_recovered = weight_recovered.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        if self.split == 0:
            return torch.nn.functional.conv2d(x, 
                                            weight_recovered,
                                            bias,
                                            self.stride, 
                                            self.padding,
                                            self.dilation,
                                            self.groups)
        else:
            weight_0_recovered = \
                self.weight_int_0.float() * self.weight_scales_0[:, None, None, None]
            weight_0_recovered = weight_0_recovered.to(x.dtype)
            output = torch.nn.functional.conv2d(x[:, :self.split, :, :], 
                                                weight_recovered,
                                                bias,
                                                self.stride, 
                                                self.padding,
                                                self.dilation,
                                                self.groups)
            output_0 = torch.nn.functional.conv2d(x[:, self.split:, :, :],
                                                weight_0_recovered,
                                                None,
                                                self.stride,
                                                self.padding,
                                                self.dilation,
                                                self.groups)
            output = output + output_0
            return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, 
                            self.dilation, self.groups)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if self.split == 0:
            x_int = quant_op(x, 
                                        self.act_scales_inv, 
                                        self.act_zero_points)
            output = qconv2d(x_int,                             # input_int
                             self.weight_int,                   # weight_int,
                             self.weight_scales,                # weight_scale
                             self.act_scales,                   # input_scale
                             self.act_zero_points,              # input_zp
                             self.scale,                        # scale
                             self.weight_sum_by_input_channels, 
                                                # weight_sum_by_input_channels
                             self.bias0,
                             self.bias,                         # bias
                             self.stride[0],                    # stride
                             self.padding[0],                   # padding
                             )
            return output
        else:
            x_int = quant_op(x[:, :self.split, :, :], 
                                        self.act_scales_inv,
                                        self.act_zero_points)
            x_int_0 = quant_op(x[:, self.split:, :, :], 
                                          self.act_scales_inv_0,
                                          self.act_zero_points_0)
            output = qconv2d(x_int,                             # input_int
                             self.weight_int,                   # weight_int,
                             self.weight_scales,                # weight_scale
                             self.act_scales,                   # input_scale
                             self.act_zero_points,              # input_zp
                             self.scale,                        # scale
                             self.weight_sum_by_input_channels, 
                                                # weight_sum_by_input_channels
                             self.bias0,
                             self.bias,                         # bias
                             self.stride[0],                    # stride
                             self.padding[0],                   # padding
                             )
            output_0 = qconv2d(x_int_0,                             # input_int
                               self.weight_int_0,                   # weight_int,
                               self.weight_scales_0,                # weight_scale
                               self.act_scales_0,                   # input_scale
                               self.act_zero_points_0,              # input_zp
                               self.scale_0,                        # scale
                               self.weight_sum_by_input_channels_0, 
                                                # weight_sum_by_input_channels
                               self.bias0_0,
                               None,            # bias. Here is none because bias
                                        # need to be applied just once in output
                               self.stride[0],                      # stride
                               self.padding[0],                     # padding
                               )
            output = output + output_0
            return output
