from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="FragenAntwortLLMCPU",
    version="0.1.19", 
    packages=find_packages(),
    install_requires=[
        "PyMuPDF",
        "tokenizers",
        "semantic-text-splitter",
        "langchain",
        "langchain-community",
        "ctransformers",
        "torch",
    ],
    author="Mehrdad Almasi, Demival Vasques, and Lars Wieneke",
    description="A package for processing documents and generating questions and answers using LLMs on CPU.",
    long_description=long_description,                      
    long_description_content_type="text/markdown",          
    python_requires=">=3.8",
)
