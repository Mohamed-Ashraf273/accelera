import subprocess

from accelera.src.utils.accelera_utils import print_msg


def clean_response(response: str) -> str:
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


def format_cpp_file(filename: str) -> bool:
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
        print_msg("Install with: sudo apt install clang-format (Ubuntu/Debian)")
        print_msg("Or: brew install clang-format (macOS)")
        return False


def convert_to_cpp(filename: str, python_code: str, ctc_chain) -> str:
    assert filename.endswith(".cpp"), "Filename must have a .cpp extension"
    response = ctc_chain.invoke({"code": python_code})
    if isinstance(response, str):
        response = response.strip()
    elif hasattr(response, "content"):
        response = response.content.strip()
    else:
        try:
            response = str(response).strip()
        except Exception:
            raise ValueError("Unable to convert model response to string.")
    cleaned_response = clean_response(response)

    with open(filename, "w") as f:
        f.write(cleaned_response)

    format_cpp_file(filename)
    return cleaned_response
