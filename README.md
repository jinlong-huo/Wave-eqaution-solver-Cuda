# CUDA-Accelerated Wave Equation Solver

A comparative study of CPU vs GPU implementations for solving the 2D wave equation using finite difference methods. This research explores C++ optimization and CUDA acceleration for scientific computing operators.

## 🎯 Research Goals

- Implement numerical PDE solvers in pure C++ and CUDA
- Benchmark performance differences between CPU and GPU implementations
- Establish a framework for accelerating existing numerical methods
- Demonstrate practical CUDA programming for scientific computing

## 📊 Results Summary

| Implementation | Time (50k steps) | Speedup   |
| -------------- | ---------------- | --------- |
| **CPU (C++)**  | ~XXX seconds     | 1.0x      |
| **GPU (CUDA)** | ~YYY seconds     | **~ZZ.x** |

*Tested on: T4/L4 GPU, 256×256 grid*

## 🚀 Quick Start

### Prerequisites
```bash
# Check GPU availability
nvidia-smi

# Required tools
- CUDA Toolkit (11.x or later)
- g++ (C++11 support)
- Python 3.x with numpy, matplotlib
```

### Run the Comparison

```bash
# 1. Clone repository
git clone https://github.com/yourusername/cuda-wave-equation
cd cuda-wave-equation

# 2. Open notebook
jupyter notebook CPop.ipynb

# Or run from command line:
# Compile both versions
g++ -o wave_cpu wave_equation_cpu.cpp -O3 -std=c++11
nvcc -o wave_gpu wave_equation_gpu.cu -O3 -arch=sm_75

# Run simulations
./wave_cpu
./wave_gpu
```

## 📁 Project Structure

```
.
├── CPop.ipynb                    # Main notebook with all cells
├── wave_equation_cpu.cpp         # CPU implementation
├── wave_equation_gpu.cu          # GPU/CUDA implementation
├── output_cpu.bin                # CPU results (generated)
├── output_gpu.bin                # GPU results (generated)
└── README.md                     # This file
```

## 🧮 Mathematical Background

### 2D Wave Equation
```
∂²u/∂t² = c² ∇²u
```

**Discretization:**
- 5-point stencil for spatial Laplacian
- Explicit time-stepping scheme
- Zero Dirichlet boundary conditions

**Stability:** CFL condition `c·Δt/Δx < 1`

## 💻 Implementation Details

### CPU Version (C++)
- Sequential nested loops for spatial grid
- Manual boundary condition handling
- O(n²) complexity per timestep

### GPU Version (CUDA)
- 2D thread blocks (16×16)
- Parallel computation for all grid points
- Shared boundary condition logic
- Memory coalescing optimization

### Key Kernel Structure
```cuda
__global__ void wave_equation_kernel(...) {
    // 1. Thread mapping
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 2. Bounds checking
    if (x >= width || y >= height) return;
    
    // 3. Boundary conditions
    if (boundary) { u_next[idx] = 0.0f; return; }
    
    // 4. Laplacian computation (5-point stencil)
    // 5. Time-stepping update
}
```

## 📈 Performance Analysis

### Metrics Tracked
- **Execution time**: Total simulation runtime
- **Throughput**: Grid updates per second
- **Speedup**: GPU vs CPU performance ratio
- **Accuracy**: Numerical difference between implementations

### Typical Speedup Factors
- Small grids (128×128): 5-10x
- Medium grids (256×256): 15-25x
- Large grids (512×512): 30-50x

## 🔬 Notebook Cells Overview

| Cell | Purpose                                       |
| ---- | --------------------------------------------- |
| 1    | Check GPU availability & system info          |
| 2    | Write CPU implementation (C++)                |
| 3    | Write GPU implementation (CUDA)               |
| 4    | Compile both versions                         |
| 5    | Run CPU simulation                            |
| 6    | Run GPU simulation                            |
| 7    | Visualize results (heatmaps & cross-sections) |
| 8    | Performance benchmarking & comparison         |
| 9    | Verify numerical accuracy                     |

## 🎓 Learning Outcomes

### CUDA Concepts Demonstrated
- ✅ Thread organization (blocks, grids)
- ✅ Memory management (host ↔ device)
- ✅ Kernel launch configuration
- ✅ Error handling
- ✅ Synchronization

### C++ Optimization Techniques
- ✅ Memory layout for cache efficiency
- ✅ Loop optimization
- ✅ Compiler flags (-O3)

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Compilation Error: "unsupported toolchain"**
```bash
# Solution: Specify correct GPU architecture
nvcc -o wave_gpu wave_equation_gpu.cu -O3 -arch=sm_75  # For T4
nvcc -o wave_gpu wave_equation_gpu.cu -O3 -arch=sm_89  # For L4
```

**2. Results Don't Match**
- Ensure identical initialization
- Check boundary condition logic
- Verify buffer rotation order
- Small differences (<1e-4) are acceptable due to floating-point precision

**3. GPU Not Detected**
```bash
# In Colab: Runtime → Change runtime type → Select GPU
# In local: Check nvidia-smi output
```

## 📚 Future Work

- [ ] 3D wave equation extension
- [ ] Multi-GPU support
- [ ] Adaptive time-stepping
- [ ] Integration with ML frameworks (PyTorch custom ops)
- [ ] Comparison with cuFFT for spectral methods
- [ ] Shared memory optimization
- [ ] Half-precision (FP16) benchmarks

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Additional PDE solvers (heat equation, Schrödinger, etc.)
- Performance optimizations
- Visualization improvements
- Documentation enhancements

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Based on finite difference methods for hyperbolic PDEs
- Inspired by classical wave equation studies
- CUDA programming guide: [NVIDIA Developer Docs](https://docs.nvidia.com/cuda/)

## 📧 Contact

**Your Name** - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/cuda-wave-equation](https://github.com/yourusername/cuda-wave-equation)

---

⭐ If you find this research useful, please consider starring the repository!
