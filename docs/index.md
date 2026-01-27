# Accelera Documentation

Welcome to **Accelera** - A High-Performance Machine Learning Pipeline Framework that combines the flexibility of Python with the performance of C++.

## Overview

Accelera is a cutting-edge ML pipeline framework designed for building and deploying machine learning workflows with optimized performance. It provides a robust, scalable solution that leverages C++ for compute-intensive operations while maintaining an intuitive Python API.

## Key Features

### Core Pipeline Module (Available)
- **Graph-Based Architecture**: Build ML pipelines as directed acyclic graphs
- **Fast Execution**: C++ backend for compute-intensive operations
- **Flexible Design**: Chain preprocessing, models, and metrics easily
- **Branch Support**: Create parallel pipeline paths for experimentation

### Code Parallelizer (Under Implementation)
- **Loop Extraction**: Extract loops from C/C++ code using AST analysis
- **Parallelization Detection**: Flag parallelizable loops using trained ML models
- **OpenMP Integration**: Automatically parallelize loops using OpenMP directives
- **Performance Optimization**: Transform sequential code into parallel execution
- **Linux Support**: Full support on Linux systems

### AutoML Module (Under Implementation)
- **Automated Pipeline Generation**: Automatically find the best pipeline for your dataset
- **Hyperparameter Optimization**: Tune models automatically
- **Smart Search**: Efficient exploration of pipeline configurations
- **Production Ready**: Generate optimized pipelines ready for deployment

### Deployment Module (Under Implementation)
- **MLOps Integration**: Complete API for model deployment
- **Model Tracking**: Track model versions and performance
- **Production Pipeline**: Deploy pipelines to production environments
- **Monitoring**: Track model performance in production

### Benchmark Platform (Under Implementation)
- **Competition System**: Users compete to find the best pipeline
- **Leaderboard**: Track performance across different approaches
- **Learning Loop**: Learn from top-performing pipelines
- **Community Driven**: Share and improve ML solutions collaboratively

## Quick Links

- [Installation Guide](installation.md) - Get started with Accelera
- [Core Pipeline](core-pipeline.md) - Graph-based ML pipelines
- [Code Parallelizer](code-parallelizer.md) - Loop extraction features
- [Examples](examples.md) - Sample projects and tutorials
- [Contributing](contributing.md) - Help improve Accelera

## Platform Support

| Platform | Core Features | Code Parallelizer | Full Support |
|----------|--------------|-------------------|--------------|
| Linux    | Supported | Supported | Full |
| Windows  | Supported | Not Available | Partial |
| macOS    | Supported | Not Available | Partial |

## Requirements

- **Python**: 3.9 or higher
- **CMake**: 3.14 or higher
- **C++ Compiler**: C++20 support required
- **LLVM/Clang 14+**: Required for code parallelizer (Linux only)

## Community

- **GitHub**: [Mohamed-Ashraf273/accelera](https://github.com/Mohamed-Ashraf273/accelera)
- **Issues**: [Report bugs or request features](https://github.com/Mohamed-Ashraf273/accelera/issues)
- **License**: Apache 2.0

## Getting Help

If you encounter any issues or have questions:

1. Check module documentation ([Core Pipeline](core-pipeline.md), [Code Parallelizer](code-parallelizer.md))
2. Look at [Examples](examples.md) for usage patterns
3. Open an issue on GitHub if you find a bug
