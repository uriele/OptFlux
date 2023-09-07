module OptFlux
using ChainRulesCore
using Flux
import ChainRulesCore: rrule, NoTangent, ZeroTangent
import Flux.@functor
import Flux: onecold, @functor , glorot_normal,glorot_uniform, OneHotLike, Dense 
import Flux: params, onehot, onehotbatch
import Plots
import Flux: _match_eltype, create_bias, _size_check,_channel_in

abstract type AbstractOptLayers end
include("utilities.jl")
include("train.jl")
include("OptLayers.jl")
include("activation.jl")
include("optconv.jl")
export AbstractOptLayers
export create_confusion_matrix,visualize_confusion_matrix
export PolyDense, enforce_nonnegative!, MixedActivationDense,PositiveDense
export negative_shift, ChainRulesCore 
export PositiveConv,PositiveConvTranspose,PositiveCrossCor,PositiveDenseConv 
# Write your package code here.

end




