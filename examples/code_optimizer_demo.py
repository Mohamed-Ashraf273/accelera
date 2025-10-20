from mainera.src.utils.code_optimizer import CodeOptimizer

optimizer = CodeOptimizer()

python_code = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

cpp_code = optimizer.convert_to_cpp("converted_code.cpp", python_code)
