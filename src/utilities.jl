
function visualize_confusion_matrix(conf_matrix::AbstractMatrix, labels::AbstractVector; kwargs...)
    heatmap(
    conf_matrix[end:-1:1, :]';
    yticks=(1:length(labels), string.(labels[end:-1:1])),
    xticks=(1:length(labels), string.(labels)),
    xlabel="True Labels",
    ylabel="Predicted Labels",
    clim=(0, maximum(conf_matrix)),
    cbar=true,
    aspect_ratio=:equal,
    size=(800, 800), kwargs...)

    for i in 1:size(conf_matrix, 1)
        for j in 1:size(conf_matrix, 2)
            annotate!([(i, j, Plots.text(string(round(conf_matrix[end+1-i, j], digits=2), "%"), 10, :grey))])
        end
    end
    return current()
end

function create_confusion_matrix(model,x_predic,y_true)
    n_classes = size(y_true,1)
    y_predic = onecold(model(x_predic))
    y_true = onecold(y_true)
    conf_matrix = zeros(Int, n_classes, n_classes)
    update_conf_matrix!(conf_matrix, y_true,y_predic)
    return conf_matrix
end


function create_confusion_matrix(y_true, y_pred)    
    n_classes = size(y_true,1)
    y_predic = onecold(y_predic)
    y_true = onecold(y_true)
    conf_matrix = zeros(Int, n_classes, n_classes)
    update_conf_matrix!(conf_matrix, y_true, y_pred)
    return conf_matrix
end

function update_conf_matrix!(conf_matrix, y_true, y_pred)
    for i in 1:length(y_true)
        true_class = y_true[i]   # +1 because Julia is 1-indexed
        pred_class = y_pred[i]   # +1 because Julia is 1-indexed
        conf_matrix[true_class, pred_class] += 1
    end
end


oftf(x,a)= convert(eltype(x),a)

nparams(layer::AbstractOptLayers) = length(layer.weight) + ((layer.bias==false) ? 0 : length(layer.bias))
