from distutils.core import setup, Extension
import numpy
import os

if os.name == 'nt':
    libraries = ['opencv_world341.lib']
    library_dirs = os.environ['PATH'].split(';')
else:
    libraries = ['opencv_core','opencv_imgproc']
    library_dirs = os.environ['LD_LIBRARY_PATH'].split(':')

wormneuronsegmentation_c = Extension('wormneuronsegmentation._wormneuronsegmentation_c',
                    sources = ['neuronsegmentation_c/neuronsegmentation.cpp','wormneuronsegmentation/_wormneuronsegmentation_c.cpp'],
                    include_dirs = [numpy.get_include()],
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args=['-O3'],)

setup (name = 'wormneuronsegmentation',
       version = '1.0',
       author='Francesco Randi @ Leiferlab, Princeton Physics',
       author_email='francesco.randi@gmail.com',
       description = 'Optimized segmentation code to locate the nuclei of neurons in stacks of fluorescence images of the worm\'s brain',
       py_modules = ['wormneuronsegmentation._wormneuronsegmentation_py'],
       ext_modules = [wormneuronsegmentation_c])
