from setuptools import setup, find_packages

setup(
    name='FragenAntwortLLMCPU',
    version='0.1.17',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF',  # for fitz
        'tokenizers',
        'semantic-text-splitter',  # Removed the specific version to allow for flexibility
        'langchain',
        'langchain_community',
        'torch',
        'torchvision',
        'torchaudio',
        'ctransformers'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cpu'
    ],
    entry_points={
        'console_scripts': [
            'process_document=FragenAntwortLLMCPU.document_processor:main',
        ],
    },
    author='Mehrdad Almasi, Demival VASQUES FILHO, and Lars Wieneke',
    author_email='Mehrdad.al.2023@gmail.com, demival.vasques@uni.lu, lars.wieneke@gmail.com',
    description='A package for processing documents and generating questions and answers using LLMs on CPU.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/FragenAntwortLLMCPU',  # Update with your actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
