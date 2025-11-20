import getpass
import os

from langchain_core.prompts import ChatPromptTemplate

from mainera.src.code_parallelizer.utils.code_utils import convert_to_cpp
from mainera.src.models.groq import GroqModel


class CodeParallelizer:
    def __init__(self, model=None):
        if model is None:
            if "GROQ_API_KEY" not in os.environ:
                api_key = getpass.getpass("Enter your Groq API key: ")
                os.environ["GROQ_API_KEY"] = api_key

            self.model = GroqModel(
                model_name="llama-3.1-8b-instant",
                api_key=os.environ["GROQ_API_KEY"],
            )
        else:
            self.model = model
        self.llm = self.model.llm()
        self.ctc_prompt = ChatPromptTemplate.from_template(
            "You are an expert C++ programmer. \n"
            "Convert the following Python code to equivalent C++ code.\n\n"
            "Requirements:\n"
            "- Use modern C++ (C++17 or later)\n"
            "- Include necessary headers\n"
            "- Use appropriate data types and STL containers\n"
            "- Ensure memory safety\n"
            "- Make the code compilable and efficient\n\n"
            "- No comments or explainations in the code\n\n"
            "Python code:\n```python\n{code}\n```\n\n"
            "C++ code:\n```cpp\n"
        )
        self.ctc_chain = self.ctc_prompt | self.llm

    def parallelize(self, output_file: str, python_code: str) -> str:
        cpp_code = convert_to_cpp(output_file, python_code, self.ctc_chain)
        return cpp_code


def parallelize_code(
    python_code: str,
    output_cpp_file: str = "optimized_code.cpp",
    model: GroqModel = None,
) -> str:
    optimizer = CodeParallelizer(model=model)
    cpp_code = optimizer.parallelize(output_cpp_file, python_code)
    return cpp_code
