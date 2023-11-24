__author__ = "Maxim Ziatdinov"
__copyright__ = "Copyright Maxim Ziatdinov (2020)"
__version__ = "0.3.9"
__maintainer__ = "Maxim Ziatdinov"
__email__ = "maxim.ziatdinov@ai4microcopy.com"
__date__ = "11/25/2020"

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='gpim',
        version='0.3.9',
        description='Gaussian processes for image analysis',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ziatdinovmax/GPim',
        author='Maxim Ziatdinov',
        author_email='maxim.ziatdinov@ai4microcopy.com',
        license='MIT license',
        packages=find_packages(include=['gpim', 'gpim.*']),
        zip_safe=False,
        install_requires=[
            'numpy>=1.16.5',
            'matplotlib>=3.1.1',
            'torch>=1.3.1',
            'pyro-ppl>=0.4.1',
            'gpytorch>=0.3.6',
            'scipy>=1.3.0'
        ],
        classifiers=['Programming Language :: Python',
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering']
    )
