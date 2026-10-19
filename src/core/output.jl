module ResultOutput

using Printf

export format_output, write_output

const MAX_INLINE_ITEMS = 12
const MAX_RECORD_PREVIEW = 5

_is_scalar(value) = value === nothing || value isa Union{Number, AbstractString, Symbol, Char}

function _write_scalar(io::IO, value)
    if value isa AbstractString
        print(io, repr(value))
    elseif value isa Float64 || value isa Float32
        @printf(io, "%.10g", value)
    else
        show(io, value)
    end
end

function _write_record(io::IO, value; depth::Int=0)
    print(io, nameof(typeof(value)))
    if depth >= 2
        return
    end

    for name in fieldnames(typeof(value))
        field = getfield(value, name)
        print(io, "\n", repeat("  ", depth + 1), name, ": ")
        _write_value(io, field; depth=depth + 1)
    end
end

function _write_value(io::IO, value; depth::Int=0)
    if _is_scalar(value)
        _write_scalar(io, value)
    elseif value isa AbstractArray
        print(io, nameof(typeof(value)), " with size ", size(value))
        if value isa Array && length(value) <= MAX_INLINE_ITEMS &&
           eltype(value) <: Number
            print(io, " = ")
            show(io, value)
        elseif value isa Array && !isempty(value) &&
               isstructtype(typeof(first(value)))
            print(io, " (preview: ")
            for (index, item) in enumerate(Iterators.take(value, MAX_RECORD_PREVIEW))
                index > 1 && print(io, "; ")
                _write_record(io, item; depth=depth + 1)
            end
            length(value) > MAX_RECORD_PREVIEW && print(io, "; ...")
            print(io, ")")
        end
    elseif value isa AbstractDict
        print(io, nameof(typeof(value)), " with ", length(value), " entries")
        if length(value) <= MAX_INLINE_ITEMS &&
           all(pair -> _is_scalar(first(pair)) && _is_scalar(last(pair)), pairs(value))
            print(io, " = ")
            show(io, value)
        end
    elseif isstructtype(typeof(value))
        _write_record(io, value; depth=depth)
    else
        show(io, value)
    end
end

function _write_orbital_energies(io::IO, energies)
    if isnothing(energies)
        print(io, "unavailable")
        return
    end

    for (index, energy) in enumerate(energies)
        index > 1 && print(io, "\n  ")
        @printf(io, "MO #%d: %.10g eV", index, energy)
    end
end

"""
    format_output(result; title=nothing) -> String

Return a readable plain-text report for a calculation result or other result
struct. Large arrays are represented by their type and dimensions to keep
reports readable, except molecular orbital energies, which are listed in eV.
"""
function format_output(result; title::Union{Nothing, AbstractString}=nothing)::String
    io = IOBuffer()
    println(io, something(title, "QuantumChemMM Calculation Report"))
    println(io, repeat("=", length(something(title, "QuantumChemMM Calculation Report"))))
    println(io, "Result type: ", typeof(result))

    if isstructtype(typeof(result))
        for name in fieldnames(typeof(result))
            print(io, name, ": ")
            value = getfield(result, name)
            if name == :orbital_energies_ev
                _write_orbital_energies(io, value)
            else
                _write_value(io, value; depth=0)
            end
            println(io)
        end
    else
        print(io, "Value: ")
        _write_value(io, result)
        println(io)
    end

    return String(take!(io))
end

"""
    write_output(filename, result; title=nothing) -> String

Write a plain-text calculation report to `filename` and return the filename.
The filename may use any extension, such as `.txt`, `.out`, or `.log`.
"""
function write_output(filename::AbstractString, result;
                      title::Union{Nothing, AbstractString}=nothing)::String
    report = format_output(result; title=title)
    open(filename, "w") do io
        write(io, report)
    end
    return String(filename)
end

end # module ResultOutput
