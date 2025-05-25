#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip> // For std::fixed and std::setprecision

// CUDA runtime API
#include <cuda_runtime.h>

// Macro to check for CUDA errors
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "           \
                      << cudaGetErrorString(err) << std::endl;                              \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// CUDA Kernel for Classical Matrix Multiplication (C = A * B)
// Assumes matrices are square (N x N)
__global__ void classicalMatrixMul(const float* A, const float* B, float* C, int N) {
    // Calculate row and column for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the matrix bounds
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA Kernel for Matrix Addition (C = A + B)
__global__ void matrixAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA Kernel for Matrix Subtraction (C = A - B)
__global__ void matrixSub(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = A[idx] - B[idx];
    }
}

// Helper function to allocate and copy matrix to device
void allocateAndCopyToDevice(const std::vector<float>& hostMatrix, float** deviceMatrix, int N) {
    CUDA_CHECK(cudaMalloc(deviceMatrix, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*deviceMatrix, hostMatrix.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
}

// Helper function to copy matrix from device to host
void copyFromDeviceToHost(const float* deviceMatrix, std::vector<float>& hostMatrix, int N) {
    CUDA_CHECK(cudaMemcpy(hostMatrix.data(), deviceMatrix, N * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// Helper function to free device memory
void freeDeviceMemory(float* deviceMatrix) {
    CUDA_CHECK(cudaFree(deviceMatrix));
}

// Strassen's Matrix Multiplication (Hybrid CPU-GPU approach)
// This function will perform the recursive calls on the CPU,
// but the base case multiplications and matrix additions/subtractions
// will be offloaded to the GPU using CUDA kernels.
// N must be a power of 2 for this simplified implementation.
void strassenMatrixMul(const float* A, const float* B, float* C, int N,
                       float* d_tempA, float* d_tempB, float* d_tempC,
                       float* d_M1, float* d_M2, float* d_M3, float* d_M4,
                       float* d_M5, float* d_M6, float* d_M7) {

    // Base case for recursion: if N is small, use classical multiplication on GPU
    // The threshold can be tuned for performance.
    // For simplicity, we'll use a relatively small base case for demonstration.
    // In a real optimized implementation, this would be much larger.
    if (N <= 32) { // Base case threshold (e.g., 32x32 matrices)
        dim3 threadsPerBlock(16, 16); // Example block size
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        classicalMatrixMul<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to complete
        return;
    }

    int halfN = N / 2;
    size_t subMatrixSize = halfN * halfN * sizeof(float);

    // Allocate temporary device memory for sub-matrices if not already allocated
    // For simplicity, we assume d_tempA, d_tempB, d_tempC are pre-allocated
    // and large enough to hold N*N matrices.
    // In a true recursive implementation, these would be managed more dynamically
    // or passed down as pointers to sub-regions of larger allocated blocks.

    // Pointers for sub-matrices (these are offsets into the larger allocated d_tempA/B/C)
    // A = | A11 A12 |
    //     | A21 A22 |
    // B = | B11 B12 |
    //     | B21 B22 |
    // C = | C11 C12 |
    //     | C21 C22 |

    // We need to extract sub-matrices A11, A12, A21, A22, B11, B12, B21, B22
    // and store them in contiguous memory for kernel calls.
    // This involves copying data from the parent matrix to temporary sub-matrix buffers.
    // For a full CUDA Strassen, this partitioning would ideally be handled by kernels
    // or by clever pointer arithmetic if matrices are stored in a specific layout.
    // For this example, we'll use temporary buffers for clarity.

    // For simplicity of this example, we'll assume the input matrices A and B
    // are already structured such that their quadrants can be accessed by
    // pointer arithmetic. This is a simplification.
    // A more robust implementation would involve kernels to extract quadrants.

    // A11, A12, A21, A22, B11, B12, B21, B22 are "views" into A and B
    const float* A11 = A;
    const float* A12 = A + halfN;
    const float* A21 = A + N * halfN;
    const float* A22 = A + N * halfN + halfN;

    const float* B11 = B;
    const float* B12 = B + halfN;
    const float* B21 = B + N * halfN;
    const float* B22 = B + N * halfN + halfN;

    // Temporary matrices for Strassen's algorithm (S1-S10, P1-P7)
    // These will be allocated once at the beginning of the main function
    // and reused recursively.
    // d_tempA, d_tempB, d_tempC are general purpose temporary buffers.
    // d_M1 to d_M7 are for the 7 products.

    dim3 threadsPerBlockAddSub(256); // For addition/subtraction kernels
    dim3 numBlocksAddSub((halfN * halfN + threadsPerBlockAddSub.x - 1) / threadsPerBlockAddSub.x);

    // S1 = B12 - B22
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(B12, B22, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M1 = A11 * S1
    strassenMatrixMul(A11, d_tempA, d_M1, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S2 = A11 + A12
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(A11, A12, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M2 = S2 * B22
    strassenMatrixMul(d_tempA, B22, d_M2, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S3 = A21 + A22
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(A21, A22, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M3 = S3 * B11
    strassenMatrixMul(d_tempA, B11, d_M3, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S4 = B21 - B11
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(B21, B11, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M4 = A22 * S4
    strassenMatrixMul(A22, d_tempA, d_M4, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S5 = A11 + A22
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(A11, A22, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // S6 = B11 + B22
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(B11, B22, d_tempB, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M5 = S5 * S6
    strassenMatrixMul(d_tempA, d_tempB, d_M5, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S7 = A12 - A22
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(A12, A22, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // S8 = B21 + B22
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(B21, B22, d_tempB, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M6 = S7 * S8
    strassenMatrixMul(d_tempA, d_tempB, d_M6, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // S9 = A11 - A21
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(A11, A21, d_tempA, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // S10 = B11 + B12
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(B11, B12, d_tempB, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());
    // M7 = S9 * S10
    strassenMatrixMul(d_tempA, d_tempB, d_M7, halfN, d_tempA, d_tempB, d_tempC, d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

    // Calculate C11, C12, C21, C22
    // C11 = M5 + M4 - M2 + M6
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_M5, d_M4, d_tempA, halfN * halfN); // tempA = M5 + M4
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_tempA, d_M2, d_tempB, halfN * halfN); // tempB = tempA - M2
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_tempB, d_M6, C, halfN * halfN); // C11 = tempB + M6
    CUDA_CHECK(cudaDeviceSynchronize());

    // C12 = M1 + M2
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_M1, d_M2, C + halfN, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());

    // C21 = M3 + M4
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_M3, d_M4, C + N * halfN, halfN * halfN);
    CUDA_CHECK(cudaDeviceSynchronize());

    // C22 = M5 + M1 - M3 - M7
    matrixAdd<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_M5, d_M1, d_tempA, halfN * halfN); // tempA = M5 + M1
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_tempA, d_M3, d_tempB, halfN * halfN); // tempB = tempA - M3
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixSub<<<numBlocksAddSub, threadsPerBlockAddSub>>>(d_tempB, d_M7, C + N * halfN + halfN, halfN * halfN); // C22 = tempB - M7
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Function to initialize a matrix with random values
void initializeMatrix(std::vector<float>& matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }
}

// Function to print a matrix (for debugging small matrices)
void printMatrix(const std::vector<float>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Open a CSV file to store results
    std::ofstream outputFile("matrix_multiplication_results.csv");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening results file!" << std::endl;
        return 1;
    }
    outputFile << "MatrixSize,ClassicalTimeMs,ClassicalMemoryBytes,StrassenTimeMs,StrassenMemoryBytes\n";

    // Define matrix sizes to test (must be powers of 2 for Strassen's)
    std::vector<int> matrixSizes = {64, 128, 256, 512, 1024}; // Example sizes

    for (int N : matrixSizes) {
        std::cout << "Testing N = " << N << std::endl;

        // Allocate host memory for matrices
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C_classical(N * N);
        std::vector<float> h_C_strassen(N * N);

        // Initialize matrices
        initializeMatrix(h_A, N);
        initializeMatrix(h_B, N);

        // --- Classical Matrix Multiplication ---
        float* d_A_classical, *d_B_classical, *d_C_classical;
        
        // Measure memory usage for classical
        size_t classical_mem_bytes = 3 * N * N * sizeof(float); // A, B, C matrices

        // Allocate device memory
        allocateAndCopyToDevice(h_A, &d_A_classical, N);
        allocateAndCopyToDevice(h_B, &d_B_classical, N);
        CUDA_CHECK(cudaMalloc(&d_C_classical, N * N * sizeof(float)));

        // Create CUDA events for timing
        cudaEvent_t start_classical, stop_classical;
        CUDA_CHECK(cudaEventCreate(&start_classical));
        CUDA_CHECK(cudaEventCreate(&stop_classical));

        // Define grid and block dimensions for classical multiplication
        dim3 threadsPerBlock(16, 16); // Example block size
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Record start event
        CUDA_CHECK(cudaEventRecord(start_classical));

        // Launch classical matrix multiplication kernel
        classicalMatrixMul<<<numBlocks, threadsPerBlock>>>(d_A_classical, d_B_classical, d_C_classical, N);

        // Record stop event and synchronize
        CUDA_CHECK(cudaEventRecord(stop_classical));
        CUDA_CHECK(cudaEventSynchronize(stop_classical));

        float classical_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&classical_time_ms, start_classical, stop_classical));
        std::cout << "  Classical CUDA Time: " << classical_time_ms << " ms" << std::endl;

        // Copy result back to host (optional, for verification)
        // copyFromDeviceToHost(d_C_classical, h_C_classical, N);

        // Free device memory for classical
        freeDeviceMemory(d_A_classical);
        freeDeviceMemory(d_B_classical);
        freeDeviceMemory(d_C_classical);
        CUDA_CHECK(cudaEventDestroy(start_classical));
        CUDA_CHECK(cudaEventDestroy(stop_classical));

        // --- Strassen's Matrix Multiplication ---
        float* d_A_strassen, *d_B_strassen, *d_C_strassen;
        float* d_tempA, *d_tempB, *d_tempC; // General purpose temporaries
        float* d_M1, *d_M2, *d_M3, *d_M4, *d_M5, *d_M6, *d_M7; // For Strassen's 7 products

        // Measure memory usage for Strassen's (approximate)
        // A, B, C + 3 general temporaries + 7 product temporaries
        size_t strassen_mem_bytes = (3 + 3 + 7) * N * N * sizeof(float);

        // Allocate device memory for Strassen's
        allocateAndCopyToDevice(h_A, &d_A_strassen, N);
        allocateAndCopyToDevice(h_B, &d_B_strassen, N);
        CUDA_CHECK(cudaMalloc(&d_C_strassen, N * N * sizeof(float)));

        // Allocate temporary matrices for Strassen's
        CUDA_CHECK(cudaMalloc(&d_tempA, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tempB, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tempC, N * N * sizeof(float))); // Can be used for intermediate results

        CUDA_CHECK(cudaMalloc(&d_M1, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M2, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M3, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M4, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M5, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M6, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_M7, N * N * sizeof(float)));

        cudaEvent_t start_strassen, stop_strassen;
        CUDA_CHECK(cudaEventCreate(&start_strassen));
        CUDA_CHECK(cudaEventCreate(&stop_strassen));

        // Record start event
        CUDA_CHECK(cudaEventRecord(start_strassen));

        // Launch Strassen's matrix multiplication
        strassenMatrixMul(d_A_strassen, d_B_strassen, d_C_strassen, N,
                          d_tempA, d_tempB, d_tempC,
                          d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7);

        // Record stop event and synchronize
        CUDA_CHECK(cudaEventRecord(stop_strassen));
        CUDA_CHECK(cudaEventSynchronize(stop_strassen));

        float strassen_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&strassen_time_ms, start_strassen, stop_strassen));
        std::cout << "  Strassen CUDA Time: " << strassen_time_ms << " ms" << std::endl;

        // Copy result back to host (optional, for verification)
        // copyFromDeviceToHost(d_C_strassen, h_C_strassen, N);

        // Free device memory for Strassen's
        freeDeviceMemory(d_A_strassen);
        freeDeviceMemory(d_B_strassen);
        freeDeviceMemory(d_C_strassen);
        freeDeviceMemory(d_tempA);
        freeDeviceMemory(d_tempB);
        freeDeviceMemory(d_tempC);
        freeDeviceMemory(d_M1);
        freeDeviceMemory(d_M2);
        freeDeviceMemory(d_M3);
        freeDeviceMemory(d_M4);
        freeDeviceMemory(d_M5);
        freeDeviceMemory(d_M6);
        freeDeviceMemory(d_M7);
        CUDA_CHECK(cudaEventDestroy(start_strassen));
        CUDA_CHECK(cudaEventDestroy(stop_strassen));

        // Write results to CSV
        outputFile << N << ","
                   << classical_time_ms << ","
                   << classical_mem_bytes << ","
                   << strassen_time_ms << ","
                   << strassen_mem_bytes << "\n";
    }

    outputFile.close();
    std::cout << "Results saved to matrix_multiplication_results.csv" << std::endl;

    return 0;
}
