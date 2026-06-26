# CUDA Simulation: Learning CUDA Graphs & Kernel Fusion

This is a personal learning project developed to understand GPU computing optimizations using CUDA. The goal was to implement a basic 1D physics simulation and transition it from a naive implementation to an optimized version using CUDA Graphs.

## What I Learned & Implemented

### 1. Kernel Fusion
Initially, the simulation launched two separate kernels: one for velocity updates and one for position updates. To minimize Global Memory bandwidth bottlenecks, I fused them into a single kernel (`updatePhysics`). This reduced unnecessary read/write operations to the GPU VRAM.

### 2. CUDA Graphs
To eliminate CPU launch overhead within the simulation loop, I used `cudaStreamBeginCapture` and `cudaStreamEndCapture` to record the kernel execution graph once, then launched the instantiated executable graph (`cudaGraphLaunch`) across 100 timesteps.

### 3. Error Handling
Implemented a robust `CUDA_CHECK` macro to validate CUDA runtime API responses and catch errors early.

## How to Run ?
* Open the project in **Visual Studio** with the **CUDA C/C++** workload installed.
* Retarget the project if prompted, and run using `Local Windows Debugger` (F5).
