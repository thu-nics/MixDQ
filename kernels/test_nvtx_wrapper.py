import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def nvtx_decorator(forward_func, name=None):
    def wrapper(self, *args, **kwargs):
        if name is not None:
            name_ = name
        else:
            name_ = f"Forward {self.__class__.__name__}"
        nvtx.range_push(name_)
        result = forward_func(self, *args, **kwargs)
        nvtx.range_pop()
        return result
    return wrapper

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

torch.cuda.cudart().cudaProfilerStart()

# Initialize the module
model = MyModule()
model.to('cuda')

# Replace the forward method with the decorated version
model.forward = nvtx_decorator(model.forward.__get__(model, MyModule))
module_ = model.linear
module_.forward = nvtx_decorator(module_.forward.__get__(module_, type(module_)), name='linear_0')

# Example usage
input_tensor = torch.randn(16, 10)
input_tensor = input_tensor.cuda()
output_tensor = model(input_tensor)

torch.cuda.cudart().cudaProfilerStop()
