import fitz  # PyMuPDF for PDF processing
from tokenizers import Tokenizer  # HuggingFace tokenizers
from semantic_text_splitter import TextSplitter  # Text splitting for large documents
from langchain_community.llms import CTransformers  # Interface for large language models
from langchain.prompts import PromptTemplate  # Template for prompts
import os  # Operating system interface
from langchain.chains import LLMChain  # Chain for linking LLM and prompt
import gc  # Garbage collector interface
import torch  # PyTorch for tensor computations
import json  # JSON for data interchange format


class DocumentProcessor:
    def __init__(
        self,
        book_path,
        temp_folder,
        output_file,
        book_name,
        start_page,
        end_page,
        number_Q_A,
        target_information,
        max_new_tokens=500,
        temperature=0.1,
        context_length=1000,
        max_tokens_chunk=400,
        arbitrary_prompt="",
        model_family="mistral",   # "mistral" (default) or "qwen"
        hf_token=None,            # optional Hugging Face token
    ):
        """
        Initialize the DocumentProcessor with the given parameters.

        Parameters:
        book_path (str): Path to the directory containing the book PDF files.
        temp_folder (str): Directory to store temporary files.
        output_file (str): Path to the output JSONL file.
        book_name (str): Name of the book PDF file.
        start_page (int): Start page number for processing (0-based).
        end_page (int): End page number for processing (0-based, exclusive).
        number_Q_A (str): Number of questions and answers to generate (as a written number).
        target_information (str): Focus of the questions and answers.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for the LLM.
        context_length (int): Maximum context length for the LLM.
        max_tokens_chunk (int): Maximum tokens per text chunk.
        arbitrary_prompt (str): Custom prompt for generating questions and answers.
        model_family (str): "mistral" or "qwen".
        hf_token (str): Optional Hugging Face API token.
        """
        self.book_path = book_path
        self.temp_folder = temp_folder
        self.output_file = output_file
        self.books = {book_name: [start_page, end_page]}
        self.number_Q_A = number_Q_A
        self.target_information = target_information
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.context_length = context_length
        self.gpu_layers = 0  # CPU-focused by default
        self.max_tokens_chunk = max_tokens_chunk
        self.arbitrary_prompt = arbitrary_prompt
        self.model_family = (model_family or "mistral").lower()

        # ---- Handle Hugging Face token (NOT GitHub) ----
        if hf_token is None:
            hf_token = (
                os.getenv("HUGGINGFACEHUB_API_TOKEN")
                or os.getenv("HF_TOKEN")
            )

        if not hf_token:
            answer = input(
                "Do you want to set a Hugging Face token (for private/gated models)? [y/N]: "
            ).strip().lower()
            if answer == "y":
                hf_token = input("Paste your Hugging Face token: ").strip()

        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # Default prompt if arbitrary_prompt is not provided
        if len(self.arbitrary_prompt) > 10:
            self.question_p = self.arbitrary_prompt
        else:
            self.question_p = (
                f"Generate up to {self.number_Q_A} sets of complex questions and their "
                f"corresponding answers from the provided text. The questions and answers "
                f"should focus on {self.target_information} and must be based directly on the "
                f"input text. Avoid questions about the main idea or purpose of the text. "
                f"Do not use pronouns or phrases like 'this period' and 'in this text'. "
                f"Complex questions should require answers that involve two or more steps. "
                f"Use specific names and terms for people, locations, agreements, dates, "
                f"events, and {self.target_information} instead of pronouns such as 'he', "
                f"'she', 'they', 'him', 'her', 'them', 'this', 'these', or 'those'. "
                f"Each question must start with a number (e.g., '1. What is ...'). Provide "
                f"each question immediately followed by its answer."
            )

        # Mapping written numbers to their numeric equivalents
        self.written_numbers = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty", "twenty-one"
        ]
        self.number_dict = {self.written_numbers[i]: i + 1 for i in range(len(self.written_numbers))}

        # Valid numbers for generating Q&A sets
        if self.number_Q_A not in self.number_dict:
            raise ValueError(
                f"number_Q_A must be one of {list(self.number_dict.keys())}, "
                f"but got: {self.number_Q_A!r}"
            )
        self.valid_number = [
            index for index in range(1, self.number_dict[self.number_Q_A] + 1)
        ]

        # Configuration for the LLM
        self.config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers,
        }

        # Prompt template for LLM
        self.template = """<s>[INST] You are a helpful, respectful and honest assistant.
Answer exactly from the context.

Context:
{context}

Instruction:
{question} [/INST] </s>"""

        # Initialize tokenizer and text splitter
        self.tokenizer22 = Tokenizer.from_pretrained("bert-base-uncased")
        self.splitter = TextSplitter.from_huggingface_tokenizer(
            self.tokenizer22,
            self.max_tokens_chunk,
        )

        # List to store all prompts
        self.all_prompts = []

    def _load_llm(self):
        """
        Load the LLM according to the selected model family (Mistral or Qwen).
        Make sure the GGUF files are downloaded and paths/filenames below match them.
        """
        print(f"Loading model family: {self.model_family}")

        if self.model_family == "mistral":
            # Newer Mistral v0.3 GGUF (example filename; adjust to what you download)
            return CTransformers(
                model="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                model_file="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                config=self.config,
                threads=os.cpu_count(),
            )

        elif self.model_family == "qwen":
            # Qwen 1.5 7B Chat GGUF (example filename; adjust to what you download)
            return CTransformers(
                model="Qwen/Qwen1.5-7B-Chat-GGUF",
                model_file="Qwen1.5-7B-Chat-Q4_K_M.gguf",
                config=self.config,
                threads=os.cpu_count(),
            )

        else:
            raise ValueError(f"Unknown model_family: {self.model_family!r}")

    def process_book(self):
        """
        Process the book to generate Q&A pairs.

        This method extracts text from the specified pages of the book, splits the text into chunks,
        and generates Q&A pairs using the LLM.
        """
        print("Working directory:", self.temp_folder)
        key = list(self.books.keys())[0]

        os.makedirs(self.temp_folder, exist_ok=True)

        qa_path = os.path.join(self.temp_folder, key + "_QA.txt")
        book_pdf_path = os.path.join(self.book_path, key)

        with open(qa_path, "w", encoding="utf-8") as prompts_write:
            count = 0
            with fitz.open(book_pdf_path) as doc:
                whole_document = ""
                # Extract text from specified pages
                for page in doc:
                    if count >= self.books[key][0] and count < self.books[key][1]:
                        text_blocks = page.get_text("blocks")
                        for block in text_blocks:
                            whole_document += (
                                block[-3]
                                .replace("¬\n", "")
                                .replace("¬ \n", "")
                            )
                    count += 1

                # Split text into chunks
                chunks = self.splitter.chunks(whole_document.strip())

            print("Number of chunks:", len(chunks))

            # Initialize LLM
            llm = self._load_llm()

            count = 0
            for context in chunks:
                prompt = PromptTemplate(
                    template=self.template,
                    input_variables=["question", "context"],
                )
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run(
                    {"question": self.question_p, "context": context}
                )

                # Write Q&A pairs to file
                line = (
                    "book_name: "
                    + key
                    + " Chunk: "
                    + context.replace("\n", " ")
                    + " \n "
                    + response
                )
                prompts_write.write(line)
                count += 1
                prompts_write.flush()
                prompts_write.write("\n******************************\n")

                print(line)
                print("Chunk", str(count), "***done***")
                print("*******************************")

            # Clean up to free memory
            print("'process_book =_= done '")
            try:
                del llm
            except Exception:
                pass
            try:
                del response
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_prompts(self):
        """
        Generate prompts from the processed Q&A pairs.

        This method reads the generated Q&A pairs, formats them, and stores them in the all_prompts list.
        """
        Chunks = []
        Begin = "<s> ### Question: { "
        key = list(self.books.keys())[0]
        qa_path = os.path.join(self.temp_folder, key + "_QA.txt")

        with open(qa_path, "r", encoding="utf-8") as reader:
            flag = True
            Allow = False
            new_line = ""
            for line in reader:
                if "****" not in line and line.rstrip().lstrip().strip() != "\n":
                    if flag:
                        book_name1 = (
                            " the books' name is "
                            + line.split("Chunk:")[0]
                            .split("book_name:")[1]
                            .replace(".pdf", "")
                            .rstrip()
                            .lstrip()
                            .strip()
                        )
                        Chunks.append(line.split("Chunk:")[1])
                        flag = False
                    if "?" in line:
                        for tt in self.valid_number:
                            if str(tt) + ". " in line:
                                Allow = True
                                break
                            else:
                                Allow = False
                        if Allow:
                            # If there is already a new_line, append it to all_prompts before starting a new one
                            if new_line:
                                self.all_prompts.append(
                                    new_line.rstrip()
                                    .lstrip()
                                    .strip()
                                    .replace("\n", "")
                                    + "} </s> \n \n "
                                )
                            clean_line = (
                                line.replace("\n", "")
                                .replace("1.", "")
                                .replace("2.", "")
                                .replace("3.", "")
                                .replace("4.", "")
                                .replace("5.", "")
                                .replace("6.", "")
                                .replace("7.", "")
                                .replace("8.", "")
                                .replace("9.", "")
                                .replace("10.", "")
                                .replace("11.", "")
                                .replace("12.", "")
                            )
                            new_line = (
                                Begin
                                + clean_line
                                + book_name1
                                + " } ### Answer: { "
                            )
                    else:
                        if Allow:
                            new_line = new_line + line.replace("Answer:", "")
                else:
                    if line.rstrip().lstrip().strip() != "\n":
                        if new_line:
                            self.all_prompts.append(
                                new_line.rstrip()
                                .lstrip()
                                .strip()
                                .replace("\n", "")
                                + "} </s> \n \n "
                            )
                        Allow = False
                        flag = True
                        new_line = ""

        # Ensure the last prompt is added if there is any
        if new_line:
            self.all_prompts.append(
                new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n "
            )

        print("*************************************")
        for each_prompt in self.all_prompts:
            print(each_prompt)
            print("*************************************")
        print("*** done ***")

    def save_to_jsonl(self):
        """
        Save the generated prompts to a JSONL file.

        This method writes the formatted prompts to the specified output JSONL file.
        """
        with open(self.output_file, "w", encoding="latin1") as output_jsonl_file:
            for item in self.all_prompts:
                json_object = {"text": item}
                output_jsonl_file.write(json.dumps(json_object) + "\n")
        print("******************************")
        print("The jsonl file is available:", self.output_file)
        print("******************************")


# Example usage
if __name__ == "__main__":
    # Example paths – replace with your real paths and parameters
    book_path = "path/to/your/document/"      # Directory path containing the PDF
    temp_folder = "path/to/temp/folder"
    output_file = "path/to/output/QA.jsonl"
    book_name = "YourDocument.pdf"
    start_page = 1
    end_page = 2
    number_Q_A = "five"  # written number like "one", "two", etc.
    target_information = "specific information you need"
    max_new_tokens = 1000
    temperature = 0.1
    context_length = 2100
    max_tokens_chunk = 800
    arbitrary_prompt = ""
    model_family = "mistral"  # or "qwen"

    print("started")
    processor = DocumentProcessor(
        book_path=book_path,
        temp_folder=temp_folder,
        output_file=output_file,
        book_name=book_name,
        start_page=start_page - 1,
        end_page=end_page - 1,
        number_Q_A=number_Q_A,
        target_information=target_information,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context_length=context_length,
        max_tokens_chunk=max_tokens_chunk,
        arbitrary_prompt=arbitrary_prompt,
        model_family=model_family,
    )
    print("process_book")
    processor.process_book()
    processor.generate_prompts()
    processor.save_to_jsonl()
