# ==============================================================================
# QuantumChemMM.jl: Unified Multi-Vendor GPU Device Abstraction Layer
# Supports: Nvidia (CUDA), AMD (ROCm), Intel (oneAPI), and CPU fallback
# ==============================================================================

module GPUInterface

using Printf

export AbstractDevice, CPUDevice, CUDADevice, ROCmDevice, OneAPIDevice,
       device_name, is_gpu, vendor, detect_available_devices,
       get_default_device, set_default_device!,
       to_device, to_host, synchronize_device

abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end

struct CUDADevice <: AbstractDevice
    device_id::Int
end
CUDADevice() = CUDADevice(0)

struct ROCmDevice <: AbstractDevice
    device_id::Int
end
ROCmDevice() = ROCmDevice(0)

struct OneAPIDevice <: AbstractDevice
    device_id::Int
end
OneAPIDevice() = OneAPIDevice(0)

device_name(::CPUDevice) = "Host CPU (Multithreaded SIMD)"
device_name(d::CUDADevice) = "NVIDIA CUDA GPU (Device #$(d.device_id))"
device_name(d::ROCmDevice) = "AMD ROCm GPU (Device #$(d.device_id))"
device_name(d::OneAPIDevice) = "Intel oneAPI / Level Zero GPU (Device #$(d.device_id))"

is_gpu(::CPUDevice) = false
is_gpu(::CUDADevice) = true
is_gpu(::ROCmDevice) = true
is_gpu(::OneAPIDevice) = true

vendor(::CPUDevice) = :cpu
vendor(::CUDADevice) = :nvidia
vendor(::ROCmDevice) = :amd
vendor(::OneAPIDevice) = :intel

# Global active device reference
const CURRENT_DEVICE = Ref{AbstractDevice}(CPUDevice())

get_default_device() = CURRENT_DEVICE[]

function set_default_device!(dev::AbstractDevice)
    CURRENT_DEVICE[] = dev
    @info "Active computational device set to: $(device_name(dev))"
    return dev
end

"""
    detect_available_devices()::Vector{AbstractDevice}
Scans the environment for available accelerator backends (NVIDIA, AMD, Intel, CPU).
"""
function detect_available_devices()::Vector{AbstractDevice}
    devices = AbstractDevice[CPUDevice()]

    # Check NVIDIA CUDA
    if isdefined(Main, :CUDA) && getfield(Main, :CUDA).functional()
        push!(devices, CUDADevice(0))
    end

    # Check AMD ROCm
    if isdefined(Main, :AMDGPU) && getfield(Main, :AMDGPU).functional()
        push!(devices, ROCmDevice(0))
    end

    # Check Intel oneAPI
    if isdefined(Main, :oneAPI) && getfield(Main, :oneAPI).functional()
        push!(devices, OneAPIDevice(0))
    end

    return devices
end

# Default array transfers (CPU identity, specialized by GPU extensions)
to_device(x::AbstractArray, ::CPUDevice) = x
to_host(x::AbstractArray) = Array(x)
synchronize_device(::CPUDevice) = nothing

end # module GPUInterface

