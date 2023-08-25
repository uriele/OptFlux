module OptFlux
import ChainRulesCore
import ChainRulesCore: rrule, NoTangent, ZeroTangent
import Flux.@functor
import Flux: onecold, @functor , glorot_normal,glorot_uniform, OneHotLike, Dense 
import Flux: params, onehot, onehotbatch
import Plots
import Flux: _match_eltype, create_bias, _size_check
import Flux

abstract type AbstractOptLayers end
include("utilities.jl")
include("train.jl")
include("OptLayers.jl")
include("activation.jl")
export AbstractOptLayers
export confusion_matrix,visualize_confusion_matrix
export PolyDense, enforce_nonnegative!, MixedActivationDense
export negative_shift, ChainRulesCore 
# Write your package code here.

end

