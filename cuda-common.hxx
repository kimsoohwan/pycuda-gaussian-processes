#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

// stackoverflow 14038589
#define CUDA_CALL(invocation) { _abort_on_error((invocation), __FILE__, __LINE__); }
inline void _abort_on_error(cudaError_t code, const char *filename, int line) {
    if(code != cudaSuccess) {
        std::cerr << filename << ":" << line << " cuda error: " << cudaGetErrorString(code) << "\n";
        exit(1);
    }
}
