from setuptools import setup, find_packages

setup(
    name="wikipedia-lm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)