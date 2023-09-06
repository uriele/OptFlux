using Flux, Statistics
using MLDatasets

using MLUtils 
using Plots
using ImageShow
using OptFlux
# Define custom activation functions
σ1(x) = relu(abs(x)-0.5)  # Squaring assumes that input is already positive
σ2(x) = x   # Identity function
σ3(x) = abs(x)  # Absolute function

c=Conv((3,3),1=>1,pad=(1,1),stride=(1,1))

z[1:end÷2]=W[1:end÷2,:]*x.^2+b
z[end÷2+1:end]=W[end÷2+1:end,:]*x+b

# Load MNIST data
trainset= FashionMNIST(split=:train)
testset = FashionMNIST(split=:test)
trainset= MNIST(split=:train)
testset = MNIST(split=:test)
#trainset= CIFAR10(split=:train)
#testset = MNIST(split=:test)
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
label=trainset.metadata["class_names"]
# One-hot-encode the labels
y_train = onehotbatch(y_train, 0:9)
y_test = onehotbatch(y_test, 0:9)
y_test_conv = onehotbatch(y_test_conv, 0:9)
y_train_conv = onehotbatch(y_train_conv, 0:9)
batch = 512
train_data =DataLoader((X_train, y_train),batchsize=batch, shuffle=true)
test_data = DataLoader((X_test, y_test),batchsize=batch, shuffle=true)   
train_conv_data =DataLoader((data=X_train_conv, label=y_train_conv),batchsize=batch, shuffle=true)
test_conv_data = DataLoader((data=X_test_conv, label=y_test_conv),batchsize=batch, shuffle=true)   
# Define a simple multi-layer perceptron (MLP)
model_linear = Chain(
  Dense(28^2, 128),#, relu),
  Dense(128, 64),#, relu),
  Dense(64, 10),
  softmax
)
#model_linear=deepcopy(model)

model = Chain(
    Dense(28^2, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 10),
    softmax
  )
  
  model_abs = Chain(
    Dense(28^2, 128, abs),
    Dense(128, 64, abs),
    Dense(64, 10, abs),
    softmax
  )

  model_tianwei = Chain(
    PolyDense(28^2=> 128,positive=true),
    PolyDense(128=> 64,positive=true),
    PolyDense(64=> 10),
    softmax
  )

  model_tianwei0 = Chain(
    PolyDense(28^2=> 128),
    PolyDense(128=> 64),
    PolyDense(64=> 10),
    softmax
  )

  
  model_p2 = Chain(
    PositiveDense(28^2=> 128,σ1),
    PositiveDense(128=> 64,σ1),
    Dense(64=> 10,σ1),
    softmax
  )

  model_2 = Chain(
    Dense(28^2=> 128,σ1),
    Dense(128=> 64,σ1),
    Dense(64=> 10,σ1),
    softmax
  )

# Loss function
loss(m,x, y) = crossentropy(m(x), y)
loss_nl(x,y)=loss(model,x,y)
loss_l(x,y)=loss(model_linear,x,y)
# Training data
# Optimizer
opt = ADAM(0.01)
# Evaluation

accuracy(m,x, y) = mean(onecold(m(x)) .== onecold(y))
# Training loop
epochs = 50
losses_nl = []
accuracies_nl = []
losses_l = []
accuracies_l = []
losses_a = []
accuracies_a = []
losses_t = []
accuracies_t = []
losses_t0 = []
accuracies_t0 = []
losses_p2 = []
accuracies_p2 = []
losses_2 = []
accuracies_2 = []

for epoch in 1:epochs
#    for batch in train_data
        Flux.train!((x,y)->loss(model_linear,x,y), Flux.params(model_linear), train_data, opt)
        Flux.train!((x,y)->loss(model,x,y), Flux.params(model), train_data, opt)
        Flux.train!((x,y)->loss(model_abs,x,y), Flux.params(model_abs), train_data, opt)
        Flux.train!((x,y)->loss(model_tianwei,x,y), Flux.params(model_tianwei), train_data, opt)
        Flux.train!((x,y)->loss(model_tianwei0,x,y), Flux.params(model_tianwei0), train_data, opt)
        Flux.train!((x,y)->loss(model_p2,x,y), Flux.params(model_p2), train_data, opt)
        Flux.train!((x,y)->loss(model_2,x,y), Flux.params(model_2), train_data, opt)
        #enforce_positive!(Flux.params(model_tianwei))
        #enforce_positive!(Flux.params(model_p2))
#    end
    @show loss(model_linear,X_train, y_train)  loss(model,X_train, y_train)
    
    push!(losses_nl, loss(model,X_train, y_train))
    push!(accuracies_nl, accuracy(model,X_train, y_train))
    
    push!(losses_l, loss(model_linear,X_train, y_train))
    push!(accuracies_l, accuracy(model_linear,X_train, y_train))

    push!(losses_a, loss(model_abs,X_train, y_train))
    push!(accuracies_a, accuracy(model_abs,X_train, y_train))

    push!(losses_t, loss(model_tianwei,X_train, y_train))
    push!(accuracies_t, accuracy(model_tianwei,X_train, y_train))

    push!(losses_t0, loss(model_tianwei0,X_train, y_train))
    push!(accuracies_t0, accuracy(model_tianwei0,X_train, y_train))
    
    push!(losses_p2, loss(model_p2,X_train, y_train))
    push!(accuracies_p2, accuracy(model_p2,X_train, y_train))
    push!(losses_2, loss(model_2,X_train, y_train))
    push!(accuracies_2, accuracy(model_2,X_train, y_train))
end
using JLD2
@save "./MNIST_nonlinear.jld2" losses_nl accuracies_nl losses_l accuracies_l losses_a accuracies_a losses_p2 accuracies_p2 losses_t accuracies_t losses_t0 accuracies_t0
plot(1:10)
plot([losses_l losses_nl losses_a losses_p2 losses_t losses_t0], ylabel="loss", xlabel="epochs", label=["linear" "nonlinear" "abs" "p2_positive+dense" "tianwei_positive+dense" "tianwei"])

plot([accuracies_l accuracies_nl accuracies_a], ylabel="accuracy",xlabel="epochs", label=["linear" "nonlinear" "abs"])
plot([accuracies_l accuracies_p2 accuracies_t], ylabel="accuracy",xlabel="epochs", label=["linear" "positive shift_relu+dense" "tianwei_positive+dense"]) 
plot([accuracies_l accuracies_2 accuracies_t0], ylabel="accuracy",xlabel="epochs", label=["linear" "shift_relu+dense" "tianwei+dense"]) 

# Compute accuracy
train_accuracy_nl = accuracy(model,X_train, y_train)
test_accuracy_nl = accuracy(model,X_test, y_test)

train_accuracy_l = accuracy(model_linear,X_train, y_train)
test_accuracy_l = accuracy(model_linear,X_test, y_test)

sum(onecold(model(X_train)) .== onecold(y_train))
println("Training accuracy: $train_accuracy")
println("Test accuracy: $test_accuracy")


mm_nl=create_confusion_matrix(model,X_train,y_train)
mm_l=create_confusion_matrix(model_linear,X_train,y_train)
mm_a=create_confusion_matrix(model_abs,X_train,y_train)
mm_t=create_confusion_matrix(model_tianwei,X_train,y_train)
mm_p2=create_confusion_matrix(model_p2,X_train,y_train)
mm_t0=create_confusion_matrix(model_tianwei0,X_train,y_train)
mm_2=create_confusion_matrix(model_2,X_train,y_train)

mm_nl=mm_nl./sum(y_train,dims=2).*100
mm_l=mm_l./sum(y_train,dims=2).*100
mm_a=mm_a./sum(y_train,dims=2).*100
mm_t=mm_t./sum(y_train,dims=2).*100
mm_p2=mm_p2./sum(y_train,dims=2).*100
mm_t0=mm_t0./sum(y_train,dims=2).*100
mm_2=mm_2./sum(y_train,dims=2).*100
visualize_confusion_matrix(mm_nl,string.(0:9);title="Nonlinear Model")
savefig("MNIST_nonlinear.png")
visualize_confusion_matrix(mm_l,string.(0:9);title="Linear Model")
savefig("MNIST_linear.png")
visualize_confusion_matrix(mm_a,string.(0:9);title="Abs Model")
savefig("MNIST_abs.png")
visualize_confusion_matrix(mm_t,string.(0:9);title="Tianwei Positive Model")
savefig("MNIST_tianwei.png")
visualize_confusion_matrix(mm_p2,string.(0:9);title="Positive Shifted ReLU Model")
savefig("MNIST_p2.png")
visualize_confusion_matrix(mm_t0,string.(0:9);title="Tianwei Model")
savefig("MNIST_tianwei0.png")
visualize_confusion_matrix(mm_2,string.(0:9);title="Shifter ReLU Model")
savefig("MNIST_2.png")


onecold(model(X_train))
onecold(y_train)