# ==============================================================================
# QuantumChemMM.jl Extension: NVIDIA CUDA GPU Acceleration
# Loaded automatically when the user executes: using CUDA
# ==============================================================================

module QuantumChemMMCUDAExt

using QuantumChemMM
using QuantumChemMM.GPUInterface
import QuantumChemMM.GPUInterface: to_device, to_host, synchronize_device

# Check if CUDA is available
if isdefined(Base, :get_extension)
    using CUDA
else
    import ..CUDA
end

function __init__()
    @info "QuantumChemMM.jl: NVIDIA CUDA extension initialized. GPU acceleration active."
end

# Specialized memory transfers for NVIDIA GPUs
to_device(x::AbstractArray, ::CUDADevice) = CUDA.CuArray(x)
to_device(x::AbstractArray, ::Type{CUDADevice}) = CUDA.CuArray(x)
to_host(x::CUDA.CuArray) = Array(x)
synchronize_device(::CUDADevice) = CUDA.synchronize()

end # module QuantumChemMMCUDAExt

