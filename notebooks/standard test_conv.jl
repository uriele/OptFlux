using Flux, Statistics
using MLDatasets

using MLUtils 
using Plots
using ImageShow
using OptFlux
using Flux: logitcrossentropy,params,onehotbatch,onecold
using Flux.Optimise: update!
# Define custom activation functions
σ1(x) = relu(abs(x)-oftype(x,7.0))  # Squaring assumes that input is already positive

trainset= CIFAR10(split=:train)
testset = CIFAR10(split=:test)
X_train,y_train_conv = trainset[:]
X_test,y_test_conv = testset[:]
X_train_conv=deepcopy(X_train)
X_test_conv=deepcopy(X_test) # |>gpu

label=trainset.metadata["class_names"]
# One-hot-encode the labels
y_test_conv = onehotbatch(y_test_conv, 0:9)
y_train_conv = onehotbatch(y_train_conv, 0:9)
batch = 32
train_conv_data =DataLoader((data=X_train_conv, label=y_train_conv),batchsize=batch, shuffle=true)
test_conv_data = DataLoader((data=X_test_conv, label=y_test_conv),batchsize=batch, shuffle=true)   
# Define a simple multi-layer perceptron (MLP)

#model_linear=deepcopy(model)
model = Chain(
  Conv((3, 3), 3=>16, pad=(1,1), relu),
  MaxPool((2,2)),
  Conv((3, 3), 16=>32, pad=(1,1), relu),
  MaxPool((2,2)),
  Conv((3, 3), 32=>64, pad=(1,1), relu),
  MaxPool((2,2)),
  flatten,
  Dense(1024, 256, relu),
  Dropout(0.5),
  Dense(256, 10),
  softmax
)

rand32(1)
model_opt = Chain(
  PositiveConv((3, 3), 3=>16, pad=(1,1), σ1,init=rand32),
  MaxPool((2,2)),
  PositiveConv((3, 3), 16=>32, pad=(1,1),  σ1,init=rand32),
  MaxPool((2,2)),
  PositiveConv((3, 3), 32=>64, pad=(1,1),  σ1,init=rand32),
  MaxPool((2,2)),
  flatten,
  Dense(1024, 256, relu),
  Dropout(0.5),
  Dense(256, 10),
  softmax
)

learning_rate = 0.001

# Evaluation
loss(x, y) = logitcrossentropy(model(x), y)
function loss_opt(x, y)
    pred = model_opt(x)
    #println("Type of pred: ", typeof(pred))
    #println("Type of y: ", typeof(y))
    return logitcrossentropy(pred, y)
end

loss_opt(X_test_conv[:,:,:,1:2],y_test_conv[:,1:2])

function debug_model(m, x)
    tmp = x
    for layer in m.layers
        tmp = layer(tmp)
        println("After layer ", typeof(layer), ", shape: ", size(tmp), ", type: ", eltype(tmp))
    end
    return tmp
end


debug_model(model_opt, X_test_conv[:,:,:,1:2])
debug_model(model, X_test_conv[:,:,:,1:2])




optimizer = ADAM(learning_rate)
accuracy(m,x, y) = mean(onecold(m(x)) .== onecold(y))

# Training loop
epochs = 5
train_epoch_losses = Float64[]
test_epoch_losses = Float64[]
train_epoch_accuracies = Float64[]
test_epoch_accuracies = Float64[]
for epoch in 1:epochs
    println("Epoch: $epoch")
    train_loss = 0.0f0
    train_correct = 0
    train_total = 0

    for (x, y) in train_conv_data
        # Forward and backward pass
        grads = gradient(params(model_opt)) do
            l = loss_opt(x, y)
            return l
        end
        l = loss_opt(x, y)
        train_loss += l
        # Update model parameters
        update!(optimizer, params(model_opt), grads)

        # Calculate training set accuracy
        pred = onecold(model_opt(x))
        train_correct += sum(pred .== onecold(y))
        train_total += length(y)
    end

    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = train_loss / length(train_conv_data)
    train_accuracy = train_correct / train_total

    # Save training loss and accuracy
    push!(train_epoch_losses, avg_train_loss)
    push!(train_epoch_accuracies, train_accuracy)

    # Evaluate model performance on test set
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    for (x, y) in test_conv_data
        l = loss_opt(x, y)
        test_loss += l

        pred = onecold(model_opt(x))
        test_correct += sum(pred .== onecold(y))
        test_total += length(y)
    end

    avg_test_loss = test_loss / length(test_conv_data)
    test_accuracy = test_correct / test_total

    # Save test loss and accuracy
    push!(test_epoch_losses, avg_test_loss)
    push!(test_epoch_accuracies, test_accuracy)

    println("Train Loss: $avg_train_loss, Train Accuracy: $train_accuracy")
    println("Test Loss: $avg_test_loss, Test Accuracy: $test_accuracy")
end


ctrain_epoch_losses = Float64[]
ctest_epoch_losses = Float64[]
ctrain_epoch_accuracies = Float64[]
ctest_epoch_accuracies = Float64[]
for epoch in 1:epochs
    println("Epoch: $epoch")
    train_loss = 0.0f0
    train_correct = 0
    train_total = 0

    for (x, y) in train_conv_data
        # Forward and backward pass
        grads = gradient(params(model)) do
            l = loss(x, y)
            return l
        end
        l = loss(x, y)
        train_loss += l
        # Update model parameters
        update!(optimizer, params(model), grads)

        # Calculate training set accuracy
        pred = onecold(model(x))
        train_correct += sum(pred .== onecold(y))
        train_total += length(y)
    end

    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = train_loss / length(train_conv_data)
    train_accuracy = train_correct / train_total

    # Save training loss and accuracy
    push!(ctrain_epoch_losses, avg_train_loss)
    push!(ctrain_epoch_accuracies, train_accuracy)

    # Evaluate model performance on test set
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    for (x, y) in test_conv_data
        l = loss(x, y)
        test_loss += l

        pred = onecold(model(x))
        test_correct += sum(pred .== onecold(y))
        test_total += length(y)
    end

    avg_test_loss = test_loss / length(test_conv_data)
    test_accuracy = test_correct / test_total

    # Save test loss and accuracy
    push!(ctest_epoch_losses, avg_test_loss)
    push!(ctest_epoch_accuracies, test_accuracy)

    println("Train Loss: $avg_train_loss, Train Accuracy: $train_accuracy")
    println("Test Loss: $avg_test_loss, Test Accuracy: $test_accuracy")
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