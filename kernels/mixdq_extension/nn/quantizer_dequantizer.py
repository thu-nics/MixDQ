import torch.nn as nn
import torch
from .utils import (quantize_from_qparams, dtype_to_bw, dequantize_to_float16)

class Quantizer(nn.Module):
    def __init__(self, qparams=None):
        super().__init__()
        self.qparams = qparams
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qparams is None or x.dtype == self.qparams.dtype:
            return x
        else:
            return quantize_from_qparams(x, self.qparams)
    
    @classmethod
    def from_float(cls, q_stub):
        return cls()
    
    def _get_name(self):
        bitwidth = 16 if self.qparams is None else \
            dtype_to_bw[self.qparams.dtype]
        return f"Quantizer({bitwidth})"

class DeQuantizer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            return x
        else:
            return dequantize_to_float16(x)
    
    @classmethod
    def from_float(cls, dq_stub):
        return cls()

    def _get_name(self):
        return f"DeQuantizer"