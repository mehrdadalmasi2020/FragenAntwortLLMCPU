# FragenAntwortLLMCPU (Generating Efficient Question and Answer Pairs for LLM Fine-Tuning)

[![Downloads](https://static.pepy.tech/badge/fragenantwortllmcpu)](https://pepy.tech/project/fragenantwortllmcpu)

Incorporating question and answer pairs is crucial for creating accurate, context-aware, and user-friendly Large Language Models (LLMs).  
**FragenAntwortLLMCPU** is a Python package for processing PDF documents and generating efficient Q&A pairs using LLMs on **CPU** only.  
These Q&A pairs can be used to fine-tune an LLM or to build specialized training datasets.

The package leverages various NLP libraries and supports multiple GGUF models, including:

- **Mistral-7B-Instruct v0.3 GGUF** (default)
- **Qwen1.5-7B-Chat GGUF**

via the `ctransformers` backend.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Selection](#model-selection)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## Installation

You can install the package from PyPI.

### Linux / macOS

```bash
pip install FragenAntwortLLMCPU
```

### Windows Users with Anaconda

Due to some package conflicts, you may need extra steps to work around `tbb`.

#### Step 1: Uninstall `tbb` manually (if previously installed)

1. Find the installation location:

   ```bash
   conda list tbb
   ```

2. Navigate to the directory shown and remove the `tbb` package files (the directory associated with `tbb`).

#### Step 2: Install `tbb` using `conda`

```bash
conda install -c conda-forge tbb
```

#### Step 3: Install FragenAntwortLLMCPU with `pip`

```bash
pip install FragenAntwortLLMCPU
```

#### Alternative: Force reinstall with pip

If you prefer not to manually remove `tbb`, you can try:

```bash
pip install --ignore-installed --force-reinstall FragenAntwortLLMCPU
```

---

## Usage

Here is an example of how to use the `DocumentProcessor`:

```python
from FragenAntwortLLMCPU import DocumentProcessor

processor = DocumentProcessor(
    book_path="/path/to/your/book/",      # Directory containing the PDF
    temp_folder="/path/to/temp/folder",
    output_file="/path/to/output/QA.jsonl",
    book_name="example.pdf",
    start_page=9,
    end_page=77,
    number_Q_A="one",                     # written number: "one", "two", ...
    target_information="foods and locations",
    max_new_tokens=1000,
    temperature=0.1,
    context_length=2100,
    max_tokens_chunk=400,
    arbitrary_prompt="",
    model_family="mistral",               # or "qwen"
    # hf_token="your_hf_token_here",      # optional, can also come from env vars
)

processor.process_book()
processor.generate_prompts()
processor.save_to_jsonl()
```

### Parameter Explanation

- **book_path**: Directory path where your PDF files are stored.
- **temp_folder**: Directory for temporary output (e.g., intermediate Q&A text files).
- **output_file**: Path to the final JSONL file containing the formatted Q&A pairs.
- **book_name**: Name of the PDF file to process.
- **start_page**: Starting page number for processing (1-based in the example).
- **end_page**: Ending page number for processing (1-based in the example).
- **number_Q_A**: Number of questions and answers to generate (as a written number, e.g. `"one"`, `"five"`).
- **target_information**: The focus of the questions and answers. You can specify domain-specific entities like  
  `"genes, diseases, locations"` or `"people, organizations, agreements"`.
- **max_new_tokens**: Maximum number of tokens to generate per response.
- **temperature**: Sampling temperature for the LLM (higher = more diverse).
- **context_length**: Maximum context length for the LLM.
- **max_tokens_chunk**: Maximum number of tokens per text chunk before sending to the LLM.
- **arbitrary_prompt**: Custom prompt to override the default question-generation instructions.
- **model_family**: Selects which underlying LLM to use. Supported values:
  - `"mistral"` (default)
  - `"qwen"`
- **hf_token** (optional): Hugging Face API token. If not provided, the package will look for
  `HUGGINGFACEHUB_API_TOKEN` or `HF_TOKEN` in the environment, and can also ask interactively.

---

## Model Selection

FragenAntwortLLMCPU uses `ctransformers` with GGUF models. You can choose the model family with the
`model_family` parameter:

- `model_family="mistral"`  
  Uses a Mistral-7B-Instruct v0.3 GGUF checkpoint.

- `model_family="qwen"`  
  Uses a Qwen1.5-7B-Chat GGUF checkpoint.

You must download the appropriate `.gguf` files and ensure their filenames and locations match the
configuration in `document_processor.py`. These files are not bundled in the Python package.

If a Hugging Face token is required (for private or gated models), you can:

- Set `HUGGINGFACEHUB_API_TOKEN` or `HF_TOKEN` in your environment, or
- Pass `hf_token="..."` to `DocumentProcessor`.

---

## Features

- Extracts text from PDF documents.
- Splits text into manageable chunks for LLM processing.
- Generates efficient question–answer pairs based on specific target information.
- Supports custom prompts for question generation.
- Runs entirely on **CPU** (no GPU required).
- Supports multiple GGUF models:
  - Mistral-7B-Instruct v0.3 (default)
  - Qwen1.5-7B-Chat
- Accepts PDF input in multiple languages (e.g. French, German, English) and generates Q&A pairs in English.

---

## Contributing

Contributions are welcome!  
Please fork the repository, open issues for bugs or feature requests, and submit pull requests with your improvements.

---

## License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## Authors

- Mehrdad Almasi
- Lars Wieneke
- Demival Vasques
