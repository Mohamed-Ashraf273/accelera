from accelera.src.code_parallelizer.utils.code_utils import get_flagger
from accelera.src.code_parallelizer.utils.code_utils import get_writer


class CodeParallelizer:
    def __init__(self):
        self.flagger = get_flagger()
        self.writer = get_writer()

    def parallelize(
        self, output_file: str, cpp_code: str, in_place: bool
    ) -> str:
        pass


def parallelize_code(
    python_code: str,
    output_cpp_file: str = "optimized_code.cpp",
    in_place: bool = False,
) -> str:
    optimizer = CodeParallelizer()
    cpp_code = optimizer.parallelize(
        output_cpp_file, python_code, in_place=in_place
    )
    return cpp_code
