
"""
    enforce_nonnegative!(params)

Enforce nonnegative parameters by setting negative values to zero. Used for inchoerent optical layers.
"""
function enforce_nonnegative!(params)
    for param in params
        param .= max.(param, 0)
    end
end

