
"""
    negative_shift(x; shift = 1.0)

Compute the negative shift of `x` along the first dimension. Used to correct optical inchoerent 
layers before applying softmax.

    negative_shift(x; shift=1.0) = x .- shift
"""
negative_shift(x::Union{T,AbstractArray{T}};shift=1.0) where {T} = x.-oftf(x,shift) 


function ∇negative_shift(dy, y) 
        tmp = dy        
end

function ChainRulesCore.rrule(::typeof(negative_shift), x; shift=1.0)
    y = negative_shift(x;shift=shift)
    negat_pullback(dy) = (NoTangent(), ∇negat(unthunk(dy), y))
    return y, negat_pullback
end
