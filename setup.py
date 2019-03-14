from distutils.core import setup, Extension
import numpy

libraries = ['opencv_core.so.3.4','opencv_imgproc.so.3.4']

# How the paths likely look like on your computer
lib_dir = '/usr/local/lib/'
include_dir = '/usr/local/include/'

extra_objects = ['{}lib{}'.format(lib_dir, l) for l in libraries]
includes = [include_dir+"opencv2"]

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
