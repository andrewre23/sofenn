from setuptools import setup, find_packages

def read_requirements(file):
    with open(file, 'r') as f:
        return [
            line.strip() for line in f.readlines()
            if line.strip() and not line.startswith('#')
        ]

# Read core and build requirements
core_requirements = read_requirements('requirements.txt')
build_requirements = read_requirements('build-requirements.txt')

setup(
    name='sofenn',
    version='0.1.3',
    author='Andrew Edmonds',
    author_email='andrewre23@gmail.com',
    description='Keras model of a Self-Organizing Fuzzy Network',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/andrewre23/sofenn',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.9',
    install_requires=core_requirements, # core
    extras_require={
        'build': build_requirements,    # build/test
    }
)
