from setuptools import setup, find_packages

NAME = 'parc'
AUTHOR = "Bonnie Li"
AUTHOR_EMAIL = "bonniesjli@gmail.com"
DESCRIPTION = 'Framework for Reinforcement Learning'
# with open('README.rst') as f:
#     LONG_DESCRIPTION = f.read()
# CLASSIFIERS = [
#     'Development Status :: 3 - Alpha',
#     'Environment :: Console',
#     'Intended Audience :: Developers',
#     'Intended Audience :: Science/Research',
#     'Intended Audience :: Education',
#     'License :: OSI Approved :: BSD License',
#     'Operating System :: OS Independent',
#     'Programming Language :: Python :: 2.7',
#     'Programming Language :: Python :: 3',
#     'Programming Language :: Python :: 3.6',
#     'Programming Language :: Python :: 3.7',
#     'Topic :: Scientific/Engineering',
#     'Topic :: Utilities',
#     'Topic :: Software Development :: Libraries',
# ]

if __name__ == '__main__':
    setup(
        name=NAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'matplotlib',
            'numpy',
            'pandas',
            'torch',
            'tensorboardX']
        )
