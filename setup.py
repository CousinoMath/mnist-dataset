from setuptools import setup, find_packages

setup(name="mnist",
    version='0.0',
    packages=find_packages(),
    install_requires=('docutils', 'numpy', 'theano'),
    package_data={
        '': ['*.txt', '*.rst', '*.gz']
    },
)