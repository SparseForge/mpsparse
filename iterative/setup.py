from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import subprocess


def compile_metal(name):
    print(f"Compiling {name}.metal...")
    os.system(f"xcrun -sdk macosx metal -c {name}.metal -o {name}.air")
    os.system(f"xcrun -sdk macosx metallib {name}.air -o {name}.metallib")
    if os.path.exists(f"{name}.air"):
        os.remove(f"{name}.air")

compile_metal("coo_csr")
compile_metal("spmv")

current_dir = os.path.dirname(os.path.abspath(__file__))
metal_cpp_path = os.path.join(current_dir, "metal-cpp")

setup(
    name='spmv',
    ext_modules=[
        CppExtension(
            name='spmv',
            sources=['spmv.cpp'],
            include_dirs=[metal_cpp_path],
            extra_compile_args=['-std=c++17'],
            extra_link_args=[
                '-framework', 'Metal', 
                '-framework', 'Foundation', 
                '-framework', 'QuartzCore'
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)