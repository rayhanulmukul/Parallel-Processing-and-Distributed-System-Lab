%%writefile matmul.cu
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int P, int offset) {
    int k = threadIdx.x + offset;

    float* a = A + k * M * N;
    float* b = B + k * N * P;
    float* c = C + k * M * P;

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            for(int l = 0; l < P; l++) {
                c[i * P + l] += a[i * N + j] * b[j * P + l];
            }
        }
    }
}

int main(int argc, char *argv[]) {

    int T = atoi(argv[1]); // threads
    int K = atoi(argv[2]); // matrices

    int M = 400, N = 400, P = 400;

    int SizeA = M * N * K;
    int SizeB = N * P * K;
    int SizeC = M * P * K;

    // memory alocate (cpu allocate)
    float *h_A = new float[SizeA];
    float *h_B = new float[SizeB];
    float *h_C = new float[SizeC];


    // malloc (gpu allocate)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, SizeA * sizeof(float));
    cudaMalloc(&d_B, SizeB * sizeof(float));
    cudaMalloc(&d_C, SizeC * sizeof(float));

    // data initialize
    for (int i = 0; i < SizeA; i++) {
        h_A[i] = rand();
    }
    for(int i = 0; i < SizeB; i++) {
        h_B[i] = rand();
    }


    // copy from host to device
    cudaMemcpy(d_A, h_A, SizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SizeB * sizeof(float), cudaMemcpyHostToDevice);

    // cuda process start
    int remaining = K;
    while(remaining){
        int batch = min(remaining, T);
        matrixMul<<<1,batch>>>(d_A, d_B, d_C, M, N, P, K-remaining);
        remaining -= batch;
    }
    cudaDeviceSynchronize();

    // let's copy back to cpu
    cudaMemcpy(h_C, d_C, SizeC * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "All operation done" << endl;
}