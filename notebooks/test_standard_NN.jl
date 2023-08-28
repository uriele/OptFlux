### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ ca91cc5d-ebca-4930-946e-a0b85b706b4c
begin
	using Pkg
	using MLUtils
	using Statistics
	using MLDatasets
	using Plots
	using ImageShow
end

# ╔═╡ 2a2aaa73-a2f4-477e-9e22-dbde1016d092
Pkg.activate("C:\\Users\\menar\\.julia\\dev\\OptFlux")

# ╔═╡ f49da650-4393-11ee-35fa-fdfb4acd9058
using Flux

# ╔═╡ a58e0096-4111-43e5-bc77-bba98517dc9f
using OptFlux

# ╔═╡ dfe42556-2d98-497a-91a3-bfd080432863
md""" Define the activation function to be used in the neural network 
"""

# ╔═╡ adfcbf11-f2e0-412b-baed-ba7b6b7e0a53
begin
	σ1(x;shift=0.5)=relu(abs(x)-shift);
	σ2(x)=abs(x);
end

# ╔═╡ 46ee1a8e-9f27-405d-900b-1c1d88c060c4
md""" Create the neural network that will be used during the training
- Linear model
```julia
linear_model=Chain(
	Dense(28^2=>128),
	Dense(128=>64),
	Dense(64=>10),
	softmax
)
```
- NonLinear model
```julia
nonlinear_model=Chain(
	Dense(28^2=>128,relu),
	Dense(128=>64,relu),
	Dense(64=>10),
	softmax
)
```
- Absolute Value
```julia
abs_model=Chain(
	Dense(28^2=>128,abs),
	Dense(128=>64,abs),
	Dense(64=>10),
	softmax
)
```
- Tianwei Model (only positive)
```julia
tianwei_model = Chain(
	PolyDense(28^2=>128),
	PolyDense(128=>64),
	PolyDense(64=>10),
	negative_shift,
	softmax
)
```
- Model P2 (only positive)
```julia
p2_model = Chain(
	PositiveDense(28^2=>128,σ1),
	PositiveDense(128=>64,σ1),
	PositiveDense(64=>10,σ1),
	negative_shift,
	softmax
)
```
- Tianwei Model (only positive) + Dense Last
```julia
tianwei_dense_model = Chain(
	PolyDense(28^2=>128;positive=true),
	PolyDense(128=>64;positive=true),
	Dense(64=>10),
	softmax
)
```
- Model P2 (only positive) + Dense Last
```julia	
p2_dense_model = Chain(
	PositiveDense(28^2=>128,σ1),
	PositiveDense(128=>64,σ1),
	Dense(64=>10),
	negative_shift,
	softmax
)
```
"""

# ╔═╡ 6f40c7b1-d576-40a1-b81f-985607a3c5c9
begin
# Linear model

linear_model=Chain(
	Dense(28^2=>128),
	Dense(128=>64),
	Dense(64=>10),
	softmax
)
# NonLinear model

nonlinear_model=Chain(
	Dense(28^2=>128,relu),
	Dense(128=>64,relu),
	Dense(64=>10),
	softmax
)
#Absolute Value
	
abs_model=Chain(
	Dense(28^2=>128,abs),
	Dense(128=>64,abs),
	Dense(64=>10),
	softmax
)
#Tianwei Model (only positive)

tianwei_model = Chain(
	PolyDense(28^2=>128;positive=true),
	PolyDense(128=>64;positive=true),
	PolyDense(64=>10;positive=true),
	negative_shift,
	softmax
)
#Model P2 (only positive)
	
p2_model = Chain(
	PositiveDense(28^2=>128,σ1),
	PositiveDense(128=>64,σ1),
	PositiveDense(64=>10,σ1),
	negative_shift,
	softmax
)
#Tianwei Model (only positive) + Dense Last

tianwei_dense_model = Chain(
	PolyDense(28^2=>128;positive=true),
	PolyDense(128=>64;positive=true),
	Dense(64=>10),
	softmax
)

	
p2_dense_model = Chain(
	PositiveDense(28^2=>128,σ1),
	PositiveDense(128=>64,σ1),
	Dense(64=>10),
	negative_shift,
	softmax
)
end

# ╔═╡ 908dc095-86c8-44dc-98e4-bc4e60821d1d
begin
# Load MNIST data
	#trainset= FashionMNIST(split=:train)
	#testset = FashionMNIST(split=:test)
	trainset= MNIST(split=:train)
	testset = MNIST(split=:test)

	X_train,y_train = trainset[:]
	X_test,y_test = testset[:]
	X_train_conv=deepcopy(X_train)
	X_test_conv=deepcopy(X_test)
	X_train_conv=reshape(X_train_conv,28,28,1,:) # |>gpu
	X_test_conv=reshape(X_test_conv,28,28,1,:)# # |>gpu
	y_test_conv=deepcopy(y_test) # |>gpu
	y_train_conv=deepcopy(y_train) # |>gpu
	X_train=Flux.flatten(X_train)
	X_test=Flux.flatten(X_test)
	
	# One-hot-encode the labels
	y_train = Flux.onehotbatch(y_train, 0:9)
	y_test = Flux.onehotbatch(y_test, 0:9)
	y_test_conv = Flux.onehotbatch(y_test_conv, 0:9)
	y_train_conv = Flux.onehotbatch(y_train_conv, 0:9)
	batch = 512
	train_data =DataLoader((X_train, y_train),batchsize=batch, shuffle=true)
	test_data = DataLoader((X_test, y_test),batchsize=batch, shuffle=true)   
	train_conv_data =DataLoader((data=X_train_conv, label=y_train_conv),batchsize=batch, shuffle=true)
	test_conv_data = DataLoader((data=X_test_conv, label=y_test_conv),batchsize=batch, shuffle=true)   

end

# ╔═╡ f13f522d-9bfe-4547-b593-c4b568eb91b2
begin
	# Create utility functions
	loss(m,x, y) = Flux.crossentropy(m(x), y)
	# Optimizer
	opt = ADAM(0.01)
	# Evaluation
	accuracy(m,x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

	accuracy(m,x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

	function training_network!(model,loss,train_data,test_data;epochs=100,opt=ADAM(0.01))
		X_test,y_test=test_data;
		losses    =[];
		accuracies=[];
		for epoch in epochs
			for batch in train_data
				Flux.train!((x,y)->loss(model,x,y),Flux.params(model),batch,opt)
			end

			loss_i=loss(model,X_test,y_test)
			accur_i=accuracy(model,X_test,y_test)
			println("$epoch : $loss_i")
			push!(losses,loss_i)
			push!(accuracies,accur_i)
		end
		return losses, accuracies
	end
	

end

# ╔═╡ b94e4c40-8d13-46ec-945e-058e39b40b8a
begin
	epochs=1
	losses=[]
	accuracies=[]
	for epoch in 1:epochs
			Flux.train!((x,y)->loss(nonlinear_model,x,y),Flux.params(nonlinear_model),train_data,opt)
			
			loss_i=loss(nonlinear_model,X_test,y_test)
			accur_i=accuracy(nonlinear_model,X_test,y_test)
			println("$epoch : $loss_i")
			push!(losses,loss_i)
			push!(accuracies,accur_i)
	end
end

# ╔═╡ 43319392-a6d5-45aa-b3fd-6ca8e427fb70

begin
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
	normalization_cm=sum(y_true;dims=2);
    println(n_classes)
	y_predic = Flux.onecold(model(x_predic))
    y_true = Flux.onecold(y_true)
    cm= create_confusion_matrix(y_true, y_predic;n_classes=n_classes)
	cm=cm./normalization_cm.*100
	return cm
end


function create_confusion_matrix(y_true, y_pred;n_classes) 
	println(n_classes)
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

end

# ╔═╡ 7bed0526-8a46-4470-9960-43e9d25316e4
begin
	cm=create_confusion_matrix(nonlinear_model,X_test,y_test)
	visualize_confusion_matrix(cm,string.(0:9))
end

# ╔═╡ 0c818baa-07b1-421f-91f5-3834e9e5796d
sum(y_test;dims=2)

# ╔═╡ Cell order:
# ╠═f49da650-4393-11ee-35fa-fdfb4acd9058
# ╠═ca91cc5d-ebca-4930-946e-a0b85b706b4c
# ╠═2a2aaa73-a2f4-477e-9e22-dbde1016d092
# ╠═a58e0096-4111-43e5-bc77-bba98517dc9f
# ╟─dfe42556-2d98-497a-91a3-bfd080432863
# ╠═adfcbf11-f2e0-412b-baed-ba7b6b7e0a53
# ╟─46ee1a8e-9f27-405d-900b-1c1d88c060c4
# ╟─6f40c7b1-d576-40a1-b81f-985607a3c5c9
# ╠═908dc095-86c8-44dc-98e4-bc4e60821d1d
# ╠═f13f522d-9bfe-4547-b593-c4b568eb91b2
# ╟─b94e4c40-8d13-46ec-945e-058e39b40b8a
# ╠═43319392-a6d5-45aa-b3fd-6ca8e427fb70
# ╠═7bed0526-8a46-4470-9960-43e9d25316e4
# ╠═0c818baa-07b1-421f-91f5-3834e9e5796d
