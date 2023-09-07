using NNlib: conv, ∇conv_data, depthwiseconv, output_size
using Flux: @nospecialize,_paddims,expand,conv_reshape_bias,SamePad,calc_padding,convfilter
using Flux: DenseConvDims,conv_dims,_print_conv_opt,_channels_in,_channels_out,∇conv_data


function create_positive_bias(weights::AbstractArray, bias::Bool, dims::Integer...; positive_bias::Bool=false)
    bias ? fill!(similar(weights, dims...), (positive_bias) ? 0 : -oftype(weights[1],0.5)) : false
  end

"""
    PositiveConv(filter, in => out, σ = identity;
         stride = 1, pad = 0, dilation = 1, groups = 1, [bias, init])

Extended convolutional layer that only allows for positive weights to be used in optical neural networks. `filter` is a tuple of integers
specifying the size of the convolutional kernel;
`in` and `out` specify the number of input and output channels.

Image data should be stored in WHCN order (width, height, channels, batch).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.
This has `N = 2` spatial dimensions, and needs a kernel size like `(5,5)`,
a 2-tuple of integers.

To take convolutions along `N` feature dimensions, this layer expects as input an array
with `ndims(x) == N+2`, where `size(x, N+1) == in` is the number of input channels,
and `size(x, ndims(x))` is (as always) the number of observations in a batch.
Then:
* `filter` should be a tuple of `N` integers.
* Keywords `stride` and `dilation` should each be either single integer,
  or a tuple with `N` integers.
* Keyword `pad` specifies the number of elements added to the borders of the data array. It can be
  - a single integer for equal padding all around,
  - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
  - a tuple of `2*N` integers, for asymmetric padding, or
  - the singleton `SamePad()`, to calculate padding such that
    `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
* Keyword `groups` is expected to be an `Int`. It specifies the number of groups
  to divide a convolution into.

Keywords to control initialization of the layer:
* `init` - Function used to generate initial weights. Defaults to `glorot_uniform`.
* `bias` - The initial bias vector is all zero by default. Trainable bias can be disabled entirely
  by setting this to `false`, or another vector can be provided such as `bias = randn(Float32, out)`.

See also [`PositiveConvTranspose`](@ref), [`PositiveDepthwiseConv`](@ref), [`CrossCor`](@ref).

# Examples
```jldoctest
julia> xs = rand32(100, 100, 3, 50); # a batch of 50 RGB images

julia> layer = PositiveConv((5,5), 3 => 7, relu; bias = false)
PositiveConv((5, 5), 3 => 7, relu, bias=false)  # 525 parameters

julia> layer(xs) |> size
(96, 96, 7, 50)

julia> PositiveConv((5,5), 3 => 7; stride = 2)(xs) |> size
(48, 48, 7, 50)

julia> PositiveConv((5,5), 3 => 7; stride = 2, pad = SamePad())(xs) |> size
(50, 50, 7, 50)

julia> PositiveConv((1,1), 3 => 7; pad = (20,10,0,0))(xs) |> size
(130, 100, 7, 50)

julia> PositiveConv((5,5), 3 => 7; stride = 2, dilation = 4)(xs) |> size
(42, 42, 7, 50)
```
"""
struct PositiveConv{N,M,F,A,V} <: AbstractOptLayers
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  groups::Int
  positive_bias::Bool
end

Flux._channels_in(l::PositiveConv) = size(l.weight, ndims(l.weight)-1) * l.groups
Flux._channels_out(l::PositiveConv) = size(l.weight, ndims(l.weight))
"""
    PositiveConv(weight::AbstractArray, [bias, activation; stride, pad, dilation])

Constructs a positive convolutional layer with the given weight and bias.
Accepts the same keywords and has the same defaults as
[`Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ; ...)`](@ref Conv).

```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(5);

julia> layer = PositiveConv(weight, bias, sigmoid)  # expects 1 spatial dimension
PositiveConv((3,), 4 => 5, σ)  # 65 parameters

julia> layer(randn(100, 4, 64)) |> size
(98, 5, 64)

julia> Flux.params(layer) |> length
2
```
"""
function PositiveConv(w::AbstractArray{T,N}, b = true, σ = identity;
              stride = 1, pad = 0, dilation = 1, groups = 1, positive_bias= false) where {T,N}

  @assert size(w, N) % groups == 0 "Output channel dimension must be divisible by groups."
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(Conv, pad, size(w)[1:N-2], dilation, stride)
  bias = create_positive_bias(w, b, size(w, N);positive_bias=positive_bias)
  (positive_bias) ? bias = abs.(bias) : bias = bias
  return PositiveConv(σ, abs.(w), bias, stride, pad, dilation, groups,positive_bias)
end

function PositiveConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
            init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
            bias = true) where N
    
  weight = convfilter(k, ch; init, groups)
  PositiveConv(abs.(weight), bias, σ; stride, pad, dilation, groups)
end

@functor PositiveConv

ChainRulesCore.@non_differentiable conv_dims(::Any, ::Any)

function (c::PositiveConv)(x::AbstractArray)
  _size_check(c, x, ndims(x)-1 => _channels_in(c))
  σ = NNlib.fast_act(c.σ, x)
  cdims = conv_dims(c, x)
  xT = _match_eltype(c, x)
  (c.positive_bias) ? bias=abs.(c.bias) : bias=c.bias
  σ.(conv(xT, c.weight, cdims) .+ conv_reshape_bias(bias,c.stride))
end


Flux.conv_dims(c::PositiveConv, x::AbstractArray) =
  DenseConvDims(x, c.weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)

function Base.show(io::IO, l::PositiveConv)
  print(io, "PositiveConv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", _channels_in(l), " => ", _channels_out(l))
  _print_conv_opt(io, l)
  print(io, ")")
end


"""
    PositiveConvTranspose(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])

Standard convolutional transpose layer. `filter` is a tuple of integers
specifying the size of the convolutional kernel, while
`in` and `out` specify the number of input and output channels.

Note that `pad=SamePad()` here tries to ensure `size(output,d) == size(x,d) * stride`.

Parameters are controlled by additional keywords, with defaults
`init=glorot_uniform` and `bias=true`.

See also [`PositiveConv`](@ref) for more detailed description of keywords.

```
"""
struct PositiveConvTranspose{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  groups::Int
  positive_bias::Bool
end

Flux._channels_in(l::PositiveConvTranspose)  = size(l.weight)[end]
Flux._channels_out(l::PositiveConvTranspose) = size(l.weight)[end-1]*l.groups

"""
    PositiveConvTranspose(weight::AbstractArray, [bias, activation; stride, pad, dilation, groups])

Constructs a PositiveConvTranspose layer with the given weight and bias.
Accepts the same keywords and has the same defaults as
[`PositiveConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ; ...)`](@ref PositiveConvTranspose).
"""
function PositiveConvTranspose(w::AbstractArray{T,N}, bias = true, σ = identity;
                      stride = 1, pad = 0, dilation = 1, groups=1, positive_bias=false) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(ConvTranspose, pad, size(w)[1:N-2], dilation, stride)
  b = create_positive_bias(w, bias, size(w, N-1) * groups;positive_bias=positive_bias)

  (positive_bias) ? b = abs.(b) : b = b

  return PositiveConvTranspose(σ, abs.(w), b, stride, pad, dilation, groups,positive_bias)
end

function PositiveConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      groups = 1,
                      bias = true,
                      positive_bias=false
                      ) where N

  weight = convfilter(k, reverse(ch); init, groups)                    
  PositiveConvTranspose(weight, bias, σ; stride, pad, dilation, groups,positive_bias)
end

@functor PositiveConvTranspose

function conv_transpose_dims(c::PositiveConvTranspose, x::AbstractArray)
  # Calculate size of "input", from ∇conv_data()'s perspective...
  combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
  I = (size(x)[1:end-2] .- 1).*c.stride .+ 1 .+ (size(c.weight)[1:end-2] .- 1).*c.dilation .- combined_pad
  C_in = size(c.weight)[end-1] * c.groups
  batch_size = size(x)[end]
  # Create DenseConvDims() that looks like the corresponding conv()
  w_size = size(c.weight)
  return DenseConvDims((I..., C_in, batch_size), w_size;
                      stride=c.stride,
                      padding=c.pad,
                      dilation=c.dilation,
                      groups=c.groups,
  )
end

ChainRulesCore.@non_differentiable conv_transpose_dims(::Any, ::Any)

function (c::PositiveConvTranspose)(x::AbstractArray)
  _size_check(c, x, ndims(x)-1 => _channels_in(c))
  σ = NNlib.fast_act(c.σ, x)
  cdims = conv_transpose_dims(c, x)
  xT = _match_eltype(c, x)
  (c.positive_bias) ? bias=abs.(c.bias) : bias=c.bias
  σ.(∇conv_data(xT, abs.(c.weight), cdims) .+ conv_reshape_bias(bias,c.stride))
end

function Base.show(io::IO, l::PositiveConvTranspose)
  print(io, "PositiveConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", _channels_in(l), " => ", _channels_out(l))
  _print_conv_opt(io, l)
  print(io, ")")
end

function Flux.calc_padding(::Type{PositiveConvTranspose}, pad::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  Flux.calc_padding(Conv, pad, k .- stride .+ 1, dilation, stride)
end

"""
    PositiveDepthwiseConv(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])
    PositiveDepthwiseConv(weight::AbstractArray, [bias, activation; stride, pad, dilation])
    
Return a depthwise convolutional layer, that is a [`Conv`](@ref) layer with number of
groups equal to the number of input channels.

See [`Conv`](@ref) for a description of the arguments.

# Examples

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # a batch of 50 RGB images

julia> layer = DepthwiseConv((5,5), 3 => 6, relu; bias=false)
Conv((5, 5), 3 => 6, relu, groups=3, bias=false)  # 150 parameters 

julia> layer(xs) |> size
(96, 96, 6, 50)

julia> DepthwiseConv((5, 5), 3 => 9, stride=2, pad=2)(xs) |> size
(50, 50, 9, 50)
```
"""
function PositiveDepthwiseConv(k::NTuple{<:Any,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; 
            stride = 1, pad = 0, dilation = 1, bias = true, init = glorot_uniform,positive_bias::Bool=false)
            PositiveConv(k, ch, σ; groups=ch.first, stride, pad, dilation, bias, init,positive_bias=positive_bias)
end

function PositiveDepthwiseConv(w::AbstractArray{T,N}, bias = true, σ = identity;
                  stride = 1, pad = 0, dilation = 1,positive_bias::Bool=false) where {T,N}
  w2 = abs.(reshape(w, size(w)[1:end-2]..., 1, :))
  PositiveConv(w2, bias, σ; groups = size(w)[end-1], stride, pad, dilation,positive_bias=positive_bias)
end


"""
    PositiveCrossCor(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])

Standard cross correlation layer. `filter` is a tuple of integers
specifying the size of the convolutional kernel;
`in` and `out` specify the number of input and output channels.

Parameters are controlled by additional keywords, with defaults
`init=glorot_uniform` and `bias=true`.

See also [`PositiveConv`](@ref) for more detailed description of keywords.

"""
struct PositiveCrossCor{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  positive_bias::Bool
end

Flux._channels_in(l::PositiveCrossCor) = size(l.weight, ndims(l.weight)-1)

"""
    CrossCor(weight::AbstractArray, [bias, activation; stride, pad, dilation])

Constructs a CrossCor layer with the given weight and bias.
Accepts the same keywords and has the same defaults as
[`CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ; ...)`](@ref CrossCor).

# Examples
```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(5);

julia> layer = CrossCor(weight, bias, relu)
CrossCor((3,), 4 => 5, relu)  # 65 parameters

julia> layer(randn(100, 4, 64)) |> size
(98, 5, 64)
```
"""
function PositiveCrossCor(w::AbstractArray{T,N}, bias = true, σ = identity;
                  stride = 1, pad = 0, dilation = 1,positive_bias::Bool=false) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(CrossCor, pad, size(w)[1:N-2], dilation, stride)
  b = create_bias(w, bias, size(w, N))
  (positive_bias) ? b = abs.(b) : b = b
  return CrossCor(σ, abs.(w), b, stride, pad, dilation,positive_bias)
end

function CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                  init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                  bias = true,positive_bias=false) where N

  weight = convfilter(k, ch, init = init)
  return CrossCor(weight, bias, σ; stride, pad, dilation,positive_bias)
end

@functor PositiveCrossCor

function crosscor(x, w, ddims::DenseConvDims)
  ddims = DenseConvDims(ddims, F=true)
  return conv(x, w, ddims)
end

crosscor_dims(c::PositiveCrossCor, x::AbstractArray) =
  DenseConvDims(x, c.weight; stride = c.stride, padding = c.pad, dilation = c.dilation)

ChainRulesCore.@non_differentiable crosscor_dims(::Any, ::Any)

function (c::PositiveCrossCor)(x::AbstractArray)
  _size_check(c, x, ndims(x)-1 => _channels_in(c))
  σ = NNlib.fast_act(c.σ, x)
  cdims = crosscor_dims(c, x)
  xT = _match_eltype(c, x)
  (c.positive_bias) ? bias=abs.(c.bias) : bias=c.bias
  σ.(crosscor(xT, abs.(c.weight), cdims) .+ conv_reshape_bias(bias,c.stride))
end

function Base.show(io::IO, l::PositiveCrossCor)
  print(io, "PositiveCrossCor(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
  _print_conv_opt(io, l)
  print(io, ")")
end
