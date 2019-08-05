from distutils.core import setup, Extension
import numpy
import os

francesco = False
tigressdata = True

libraries = ['opencv_core.so.3.4','opencv_imgproc.so.3.4']

# How the paths likely look like on your computer
if francesco: 
    lib_dir = '/usr/local/lib/'
    include_dir = '/usr/local/include/'
elif tigressdata:
    lib_dir = '/home/frandi/.local/lib64/'
    include_dir = '/home/frandi/.local/include/'
elif os.name == 'nt':
    lib_dir = 'C:\\Users\\francesco\\dev\\opencv\\build\\x64\\vc14\\lib\\'
    include_dir = 'C:\\Users\\francesco\\dev\\opencv\\build\\include\\'
    libraries = ['opencv_world341.lib']

extra_objects = ['{}lib{}'.format(lib_dir, l) for l in libraries]
if os.name == 'nt': extra_objects = ['{}{}'.format(lib_dir, l) for l in libraries]

if francesco: 
    includes = [include_dir+"opencv2"]
elif tigressdata: 
    includes = [include_dir]
elif os.name == 'nt':
    includes = [include_dir]


includes.append(numpy.get_include())

pyneuronsegmentation_c = Extension('pyneuronsegmentation._pyneuronsegmentation_c',
                    sources = ['pyneuronsegmentation/_pyneuronsegmentation_c.cpp'],
                    include_dirs = includes,
                    extra_objects=extra_objects,
                    extra_compile_args=['-O3'])

setup (name = 'pyneuronsegmentation',
       version = '1.0',
       author='Francesco Randi @ Leiferlab, Princeton Physics',
       author_email='francesco.randi@gmail.com',
       description = 'neuronsegmentation',
       py_modules = ['pyneuronsegmentation._pyneuronsegmentation_py'],
       ext_modules = [pyneuronsegmentation_c])
