import json
import os
from pathlib import Path

import requests

from accelera.src.utils.code_utils import extract_features
from accelera.src.utils.code_utils import extract_loops
from accelera.src.utils.code_utils import feature_dict_to_vector
from accelera.src.utils.code_utils import format_cpp_file
from accelera.src.utils.code_utils import generate_omp_pragma
from accelera.src.utils.code_utils import write_loops_to_json


class Parallelizer:
    def __init__(self):
        self.root_path = Path(__file__).resolve().parent.parent.parent.parent
        self.cache_dir = self.root_path / ".accelera_cache"
        self.predict_endpoint = (
            "https://mohamedahraf273-open-mp-classifier.hf.space/predict"
        )

    def _classify(self, embedding):
        response = requests.post(
            self.predict_endpoint, json={"embedding": embedding.tolist()}
        )
        result = response.json()
        return result["result"]

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
            features = extract_features(loop_code)
            embedding = feature_dict_to_vector(features)
            pred_class = self._classify(embedding)

            if pred_class != "none":
                pragma_with_loop = generate_omp_pragma(loop_code, pred_class)
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

        return


parallelizer = Parallelizer()
