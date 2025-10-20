import subprocess

from langchain_core.prompts import ChatPromptTemplate

from mainera.src.models.ollama import Ollama
from mainera.src.utils.mainera_utils import print_msg


class CodeOptimizer:
    def __init__(self, model=None):
        self.model = (
            model
            if model is not None
            else Ollama(model_name="deepseek-coder:6.7b-base-q4_K_M")
        )
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
            "Python code:\n```python\n{code}\n```\n\n"
            "C++ code:\n```cpp\n"
        )
        self.ctc_chain = self.ctc_prompt | self.llm

    def _clean_response(self, response: str) -> str:
        if "```cpp" in response:
            start = response.find("```cpp") + 6
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()

        response = response.rstrip("`").strip()

        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip() not in ["```", "``", "`", "````"]:
                cleaned_lines.append(line)
        response = "\n".join(cleaned_lines)

        unwanted_prefixes = [
            "Here's the C++ code:",
            "Here is the converted code:",
            "C++ code:",
            "```cpp",
            "```",
        ]

        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        return response

    def _format_cpp_file(self, filename: str) -> bool:
        try:
            subprocess.run(
                ["clang-format", "--version"], capture_output=True, check=True
            )

            subprocess.run(["clang-format", "-i", filename], check=True)

            print_msg(f"Successfully formatted {filename} with clang-format")
            return True

        except subprocess.CalledProcessError as e:
            print_msg(f"Error running clang-format: {e}")
            return False
        except FileNotFoundError:
            print_msg(
                "clang-format not found. "
                "Please install clang-format to auto-format C++ code."
            )
            print_msg(
                "Install with: sudo apt install clang-format (Ubuntu/Debian)"
            )
            print_msg("Or: brew install clang-format (macOS)")
            return False

    def convert_to_cpp(self, filename: str, python_code: str) -> str:
        response = self.ctc_chain.invoke({"code": python_code}).strip()

        cleaned_response = self._clean_response(response)

        with open(filename, "w") as f:
            f.write(cleaned_response)

        self._format_cpp_file(filename)
        return cleaned_response
