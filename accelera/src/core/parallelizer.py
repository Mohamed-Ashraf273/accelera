import json
import os
from pathlib import Path

import requests

from accelera.src.utils.code_utils import extract_features
from accelera.src.utils.code_utils import extract_loops
from accelera.src.utils.code_utils import format_cpp_file
from accelera.src.utils.code_utils import validate_pragma
from accelera.src.utils.code_utils import vectorize_features
from accelera.src.utils.code_utils import write_loops_to_json


class Parallelizer:
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent.parent
        self.cache_dir = self.root_path / ".accelera_cache"
        self.classifier_endpoint = (
            "https://accelera-ai-open-mp-classifier.hf.space/predict"
        )
        self.generator_endpoint = (
            "https://accelera-ai-open-mp-generator.hf.space/generate"
        )

    def _classify(self, embedding):
        try:
            response = requests.post(
                self.classifier_endpoint, json={"embedding": embedding.tolist()}
            )
            result = response.json()
            if response.status_code != 200:
                raise RuntimeError(
                    "Error while parallelizing, "
                    "in classifier with error: "
                    f"{result.get('error', 'Unknown error')}"
                )

            return result["result"]
        except Exception as e:
            raise RuntimeError(
                f"Error while parallelizing, in classifier with error: {e}"
            )

    def _generate_omp_pragma_with_loop(
        self, loop_code: str, loop_class: str
    ) -> str:
        if loop_class == "none":
            return loop_code

        payload = {
            "code_snippet": loop_code,
            "cls": loop_class,
            "max_len": 1500,
        }

        try:
            response = requests.post(
                self.generator_endpoint, json=payload, timeout=10
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Error while parallelizing, "
                    f"in generator with error: {response.text}"
                )
            pragma = response.json().get("pragma", "").strip()
            pragma = validate_pragma(pragma)
            return f"{pragma}\n{loop_code}"
        except Exception as e:
            raise RuntimeError(
                f"Error while parallelizing, in generator with error: {e}"
            )

    def parallelize(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            code = file.read()

        loops = extract_loops(file_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        json_path = (
            self.cache_dir / f"extracted_loops_{Path(file_path).stem}.json"
        )

        if not os.path.exists(json_path):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                write_loops_to_json(loops, str(json_path))
            except Exception as e:
                raise RuntimeError("Error writing JSON") from e

        with open(json_path, "r") as f:
            loops_data = json.load(f)

        code_lines = code.split("\n")

        shift = 0
        for loop in loops_data:
            loop_code = loop["code"]
            start_line = loop["start_line"]
            end_line = loop["end_line"]
            loop_type = loop["type"]

            if loop_type != "for":
                continue

            features = extract_features(loop_code)
            embedding = vectorize_features(features)
            pred_class = self._classify(embedding)

            if pred_class != "none":
                pragma_with_loop = self._generate_omp_pragma_with_loop(
                    loop_code, pred_class
                )
                new_lines = pragma_with_loop.split("\n")
                code_lines[start_line - 1 + shift : end_line + shift] = (
                    new_lines
                )
                shift += len(new_lines) - (end_line - start_line + 1)

        current_dir = Path(file_path).parent
        final_output_path = (
            current_dir / f"parallelized_{Path(file_path).stem}.c"
        )

        with open(final_output_path, "w") as file:
            file.write("\n".join(code_lines))

        format_cpp_file(final_output_path)


parallelizer = Parallelizer()
