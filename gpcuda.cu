#include "cuda-common.hxx"

typedef double real_t;

__global__ void cos_kernel(const real_t * in, real_t * out,
                           size_t size) {
    unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;
    if(ii < size) {
        out[ii] = cos(in[ii]);
    }
}

extern "C" {
    void cos_doubles(double *in_array, double *out_array, int size) {
        
        real_t *d_in, *d_out;
        
        CUDA_CALL(cudaMalloc((void **) &d_in, sizeof(real_t)*size));
        CUDA_CALL(cudaMalloc((void **) &d_out, sizeof(real_t)*size));        

        CUDA_CALL(cudaMemcpy(d_in, in_array, sizeof(real_t)*size,
                             cudaMemcpyHostToDevice));

        cos_kernel<<<ceil(size/1024.0), 1024>>>(d_in, d_out, size);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(out_array, d_out, sizeof(real_t)*size,
                             cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaFree(d_in));
        CUDA_CALL(cudaFree(d_out));
                  
    }
}
