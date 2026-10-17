# ==============================================================================
# QuantumChemMM.jl Extension: Intel oneAPI / Level Zero GPU Acceleration
# Loaded automatically when the user executes: using oneAPI
# ==============================================================================

module QuantumChemMMoneAPIExt

using QuantumChemMM
using QuantumChemMM.GPUInterface
import QuantumChemMM.GPUInterface: to_device, to_host, synchronize_device

if isdefined(Base, :get_extension)
    using oneAPI
else
    import ..oneAPI
end

function __init__()
    @info "QuantumChemMM.jl: Intel oneAPI extension initialized. GPU acceleration active."
end

# Specialized memory transfers for Intel GPUs
to_device(x::AbstractArray, ::OneAPIDevice) = oneAPI.oneArray(x)
to_device(x::AbstractArray, ::Type{OneAPIDevice}) = oneAPI.oneArray(x)
to_host(x::oneAPI.oneArray) = Array(x)
synchronize_device(::OneAPIDevice) = oneAPI.synchronize()

end # module QuantumChemMMoneAPIExt

