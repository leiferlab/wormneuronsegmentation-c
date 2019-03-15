from distutils.core import setup, Extension
import numpy

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

extra_objects = ['{}lib{}'.format(lib_dir, l) for l in libraries]
print(extra_objects)
if francesco: 
    includes = [include_dir+"opencv2"]
elif tigressdata: 
    includes = [include_dir]


includes.append(numpy.get_include())

pyns = Extension('pyns',
                    sources = ['pyns.cpp'],
                    include_dirs = includes,
                    extra_objects=extra_objects,
                    extra_compile_args=['-O3'])

setup (name = 'pyns',
       version = '1.0',
       description = 'pyns',
       ext_modules = [pyns])
