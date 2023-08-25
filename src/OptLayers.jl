

"""
    PolyDense(in => out, σ=identity; bias=true, init=glorot_uniform)
    PolyDense(W::AbstractMatrix, [bias, σ])

Create a polynomial fully connected layer, whose forward pass is given by:

    y[1:end÷2] = σ.(W[1:end÷2,:] * x.^2 .+ bias[1:end÷2])
    y[end÷2+1:end] = σ.(W[end÷2+1:end,:] * x .+ bias[end÷2+1:end])

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

!note: The `out` dimension has to be even.

Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@ref Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Examples
```jldoctest
julia> d = PolyDense(5 => 2)
Dense(5 => 2)       # 12 parameters

julia> d(rand32(5, 64)) |> size
(2, 64)

julia> d(rand32(5, 6, 4, 64)) |> size  # treated as three batch dimensions
(2, 6, 4, 64)

julia> d1 = PolyDense(ones(2, 5), false, tanh)  # using provided weight matrix
Dense(5 => 2, tanh; bias=false)  # 10 parameters

julia> d1(ones(5))
2-element Vector{Float64}:
 0.9999092042625951
 0.9999092042625951

julia> Flux.params(d1)  # no trainable bias
Params([[1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0]])
```
"""
struct PolyDense{F, M<:AbstractMatrix, B}<: AbstractOptLayers
  weight::M
  bias::B
  σ::F
  function PolyDense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    @assert(size(W, 1) % 2 == 0, "out dimension has to be even")
    b = create_bias(W, bias, size(W,1))
    new{F,M,typeof(b)}(W, b, σ)
  end
end


function PolyDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
    init = glorot_uniform, bias = true)
PolyDense(init(out, in), bias, σ)
end

@functor PolyDense

function (a::PolyDense)(x::AbstractVecOrMat)
_size_check(a, x, 1 => size(a.weight, 2))
σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
xT = _match_eltype(a, x)  # fixes Float64 input, etc.
weight=(a.weight)
bias=(a.bias)
#@info("weight: ", weight)
#@info("bias: ", bias)
#@info("poly:",weight[1:end÷2,:] * xT.^2 .+ bias[1:end÷2])
#@info("linear",weight[(end÷2)+1:end,:] * xT .+ bias[(end÷2)+1:end])
tmp=[weight[1:end÷2,:] * xT.^2; weight[end÷2+1:end,:] * xT].+ bias
return σ.(tmp)

end


function Base.show(io::IO, l::PolyDense)
print(io, "PolyDense(", size(l.weight, 2), " => ", size(l.weight, 1))
l.σ == identity || print(io, ", ", l.σ)
l.bias == false && print(io, "; bias=false")
print(io, ")")
end

"""
    MixedActivationDense{F, S, T}

A custom dense layer supporting mixed activation functions.

# Fields
- `weight::S`: The weight matrix.
- `bias::T`: The bias vector.
- `σ::F`: The activation function or vector of activation functions.

# Constructors

## MixedActivationDense(W::M, bias, σ)

### Parameters
- `W::M`: An `AbstractMatrix` for the weight matrix.
- `bias`: The bias vector.
- `σ`: A function or vector of functions for activation.

### Example

# Examples
```jldoctest
julia> d = MixedActivationDense(5 => 2, [tanh,identity])
Dense(5 => 2, Function[tanh,identity])       # 12 parameters

julia> d = MixedActivationDense(5 => 2, tanh)
Dense(5 => 2, tanh)       # 12 parameters
```
"""
struct MixedActivationDense{F, S, T}<: AbstractOptLayers
    weight::S
    bias::T
    σ::F
    
    function MixedActivationDense(W::M, bias, σ) where {M<:AbstractMatrix}
        @assert(size(W, 1) % 2 == 0, "out dimension has to be even")
        b = create_bias(W, bias, size(W,1))
        #σ=fill(σ,size(W,1))
        new{typeof(σ),M,typeof(b)}(W, b, σ)
    end
end


function MixedActivationDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
    init = glorot_uniform, bias = true)
    @assert( (isa(σ, Vector) && length(σ) == out) || !isa(σ,Array), "σ must be a vector of length out or a function")
    if isa(σ, Vector)
        return MixedActivationDense(init(out, in),bias,σ)
    else
        return Dense(init(out, in),bias,σ)
    end
end
Flux.@functor MixedActivationDense

function (a::MixedActivationDense)(x)
    W, b, σ = a.W, a.b, a.σ
    z = W * x .+ b
    for i in 1:length(z)
        z[i] = σ[i](z[i])
    end
    return z
end

function Base.show(io::IO, l::MixedActivationDense)
    print(io, "MixedActivationDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")      # ", nparams(l), " parameters")
end


