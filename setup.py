from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
import git

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)

if os.name == 'nt':
    libraries = ['opencv_world341.lib']
    library_dirs = os.environ['PATH'].split(';')
else:
    libraries = ['opencv_core','opencv_imgproc']
    library_dirs = os.environ['LD_LIBRARY_PATH'].split(':')
    
# Get git commit info to build version number/tag
repo = git.Repo('.git')
git_hash = repo.head.object.hexsha
git_url = repo.remotes.origin.url
v = repo.git.describe()
if repo.is_dirty(): v += ".dirty"

wormneuronsegmentation_c = Extension('wormneuronsegmentation._wormneuronsegmentation_c',
                    sources = ['neuronsegmentation_c/neuronsegmentation.cpp','wormneuronsegmentation/_wormneuronsegmentation_c.cpp'],
                    include_dirs = [],
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args=['-O3'],)

setup (name = 'wormneuronsegmentation',
       version = v,
       author='Francesco Randi @ Leiferlab, Princeton Physics',
       author_email='francesco.randi@gmail.com',
       description = 'Optimized segmentation code to locate the nuclei of neurons in stacks of fluorescence images of the worm\'s brain',
       py_modules = ['wormneuronsegmentation._wormneuronsegmentation_py'],
       ext_modules = [wormneuronsegmentation_c])
