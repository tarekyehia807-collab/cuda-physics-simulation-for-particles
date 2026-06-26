#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(error); \
        } \
    } while (0)

constexpr int N = 2'000'000;
constexpr float dt = 0.01f;
constexpr float G = 9.81f;
constexpr float boundary = 1000.0f;

__global__ void updatePhysics(float* pos, float* vel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        vel[idx] += G * dt;
        pos[idx] += vel[idx] * dt;
        if (pos[idx] > boundary) pos[idx] = 0.0f;
    }
}

int main() {
    float* h_pos, * h_vel;
    CUDA_CHECK(cudaMallocHost(&h_pos, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_vel, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_pos[i] = i * 0.1f;
        h_vel[i] = 0.0f;
    }

    float* d_pos, * d_vel;
    CUDA_CHECK(cudaMalloc(&d_pos, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_vel, h_vel, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    updatePhysics << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_pos, d_vel);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    for (int step = 0; step < 100; step++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_vel, d_vel, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos, d_pos, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Position[0] = " << h_pos[0] << '\n';
    std::cout << "Velocity[0] = " << h_vel[0] << '\n';

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFreeHost(h_pos));
    CUDA_CHECK(cudaFreeHost(h_vel));

    return 0;
}
