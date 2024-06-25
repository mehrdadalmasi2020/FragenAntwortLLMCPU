
# FragenAntwortLLMCPU (Generating efficient Question and Answer Pairs for LLM Fine-Tuning with FragenAntwortLLMCPU)
[![Downloads](https://static.pepy.tech/badge/fragenantwortllmcpu)](https://pepy.tech/project/fragenantwortllmcpu)

Incorporating question and answer pairs is crucial for creating accurate, context-aware, and user-friendly Large Language Models. 
FragenAntwortLLMCPU is a Python package designed for processing PDF documents and generating efficient Q&A pairs using large language models (LLMs) on CPU. 
This package can be used to fine-tune an LLM. 
It leverages various NLP libraries and the Mistral v1 LLM (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) to achieve this.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Contact](#contact)

## Installation

To install the required dependencies, follow these steps:

### For Linux Users

You can directly install the package using pip:

```bash
pip install FragenAntwortLLMCPU
```

### For Windows Users with Anaconda

Due to some package conflicts, you need to perform extra steps to successfully install `FragenAntwortLLMCPU`.

#### Step 1: Uninstall `tbb` manually (if you have installed it previously)
1. **Find the installation location:**
   Open Anaconda Prompt and run:
   ```bash
   conda list tbb
   ```
   This will show you the location where `tbb` is installed.

2. **Remove the `tbb` package:**
   Navigate to the directory shown by `conda list` and manually delete the `tbb` package files. This typically involves removing the directory associated with `tbb`.

#### Step 2: Install `tbb` using `conda`
1. **Install `tbb` with `conda`:**
   After manually removing the package, install `tbb` using `conda`:
   ```bash
   conda install -c conda-forge tbb
   ```

#### Step 3: Install your package using `pip`
Now that `tbb` is managed by `conda`, you can install your package without conflicts:

```bash
pip install FragenAntwortLLMCPU
```

#### Alternative: Force reinstall with pip
If you prefer not to manually remove the package, you can force the reinstallation using `pip`:

```bash
pip install --ignore-installed --force-reinstall FragenAntwortLLMCPU
```

## Usage
Here's an example of how to use the Document Processor:

```python
from FragenAntwortLLMCPU import DocumentProcessor

processor = DocumentProcessor(
    book_path="/path/to/your/book/",  # Directory path without ".pdf" term
    temp_folder="/path/to/temp/folder",
    output_file="/path/to/output/QA.jsonl",
    book_name="example.pdf",
    start_page=9,
    end_page=77,
    number_Q_A="one",  # This should be a written number like "one", "two", etc.
    target_information="foods and locations", 
    max_new_tokens=1000,
    temperature=0.1,
    context_length=2100,
    max_tokens_chunk=400,
    arbitrary_prompt=""
)

processor.process_book()
processor.generate_prompts()
processor.save_to_jsonl()
```

### Explanation

- **book_path**: The directory path where your PDF book files are stored.
- **temp_folder**: The directory where temporary files will be stored.
- **output_file**: The path to the output JSONL file where the Q&A pairs will be saved.
- **book_name**: The name of the book PDF file to process.
- **start_page**: The starting page number for processing.
- **end_page**: The ending page number for processing.
- **number_Q_A**: The number of questions and answers to generate (as a written number).
- **target_information**: The focus of the questions and answers. Add the types of information you want to create Question and Answer pairs for here. This can include various entities like gene names, disease names, locations, etc. For example, you might specify: "genes, diseases, locations" if you are working on a medical dataset.
- **max_new_tokens**: The maximum number of tokens to generate.
- **temperature**: The sampling temperature for the LLM.
- **context_length**: The maximum context length for the LLM.
- **max_tokens_chunk**: The maximum number of tokens per text chunk.
- **arbitrary_prompt**: A custom prompt for generating questions and answers.

### Features

- Extracts text from PDF documents
- Splits text into manageable chunks for processing
- Generates efficient question and answer pairs based on specific target information
- Supports custom prompts for question generation
- **Runs on CPU**: The code can be executed without the need for GPUs.
- **Uses Mistral Model (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)**: Utilizes the CTransformers version of the Mistral v1 model.
- **Multilingual Input**: Accepts PDF books in French, German, or English and generates Q&A pairs in English.



### Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Authors

- Mehrdad Almasi, Lars Wieneke and Demival VASQUES FILHO

### Contact

For questions or feedback, please contact **Mehrdad.al.2023@gmail.com, lars.wieneke@gmail.com, demival.vasques@uni.lu**.
