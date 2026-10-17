# ==============================================================================
# QuantumChemMM.jl Extension: AMD ROCm GPU Acceleration
# Loaded automatically when the user executes: using AMDGPU
# ==============================================================================

module QuantumChemMMAMDGPExt

using QuantumChemMM
using QuantumChemMM.GPUInterface
import QuantumChemMM.GPUInterface: to_device, to_host, synchronize_device

if isdefined(Base, :get_extension)
    using AMDGPU
else
    import ..AMDGPU
end

function __init__()
    @info "QuantumChemMM.jl: AMD ROCm / AMDGPU extension initialized. GPU acceleration active."
end

# Specialized memory transfers for AMD GPUs
to_device(x::AbstractArray, ::ROCmDevice) = AMDGPU.ROCArray(x)
to_device(x::AbstractArray, ::Type{ROCmDevice}) = AMDGPU.ROCArray(x)
to_host(x::AMDGPU.ROCArray) = Array(x)
synchronize_device(::ROCmDevice) = AMDGPU.synchronize()

end # module QuantumChemMMAMDGPExt

