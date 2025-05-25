# HPC-course-project

## Project Description
This project compares the performance of GPU implementations of CUDA matrix multiplication algorithms for 1024x124 matrices.
[See](https://github.com/ayagnych/HPC-course-project/blob/main/project_yagnych.pdf)

## Project Features
- CUDA-acceleration of both algorithms
- 1024x124 matrix support
- Accurate time measurement via CUDA Events
- Verification of the results by comparing the traces of the matrices
- Hybrid approach: the recursive Strassen algorithm with a transition to classical multiplication for submatrices of ≤64x64

## Requirements
- NVIDIA graphics card with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- The 'nvcc` compiler

## Installation and launch
1. Clone the repository:
   ```bash
   git clone https://github.com/ayagnych/HPC-course-project
   ```
2. ```bash
   nvcc matrix_mul.cu -o matrix_mul
   ./matrix_mul
   python plot_results.py
   ```
