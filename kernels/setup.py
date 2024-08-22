from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as path
import glob

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "mixdq_extension", "csrc") 
    
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"), 
                        recursive=True)
    source_cuda_rt = glob.glob(path.join(extensions_dir, "**", "*.cc"), 
                               recursive=True)
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu"), 
                            recursive=True)
    sources += source_cuda
    sources += source_cuda_rt

    cutlass_path = path.join(this_dir, "third_party", "nvidia-cutlass")
    cutlass_header_path = path.join(cutlass_path, 'include')
    cutlass_tool_header_path = \
        path.join(cutlass_path, 'tools', 'util', 'include')
    include_dirs = [cutlass_header_path, cutlass_tool_header_path]

    ext_modules = [
        CUDAExtension(
            "mixdq_extension._C",
            sorted(sources),
            extra_compile_args=['-std=c++17'],
            include_dirs=include_dirs,
        )
    ]
    return ext_modules

def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name="mixdq_extension",
    version="0.6",
    packages=find_packages(include=['mixdq_extension']),
    install_requires=fetch_requirements(),
    python_requires=">=3.7",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
