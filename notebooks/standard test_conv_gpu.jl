using Flux, Statistics, CUDA
using MLDatasets
using MLUtils 
using Plots
using ImageShow
using OptFlux
using Flux: logitcrossentropy, gpu
using Flux.Optimise: update!

# Define custom activation functions
σ1(x) = relu(abs(x)-0.5)

# Load data
trainset = CIFAR10(split=:train)
testset = CIFAR10(split=:test)
X_train, y_train_conv = trainset[:]
X_test, y_test_conv = testset[:]

# Move data to GPU
X_train_conv = X_train |> gpu
X_test_conv = X_test |> gpu

# One-hot-encode the labels
y_train_conv = onehotbatch(y_train_conv, 0:9) |> gpu
y_test_conv = onehotbatch(y_test_conv, 0:9) |> gpu

# Create DataLoader
batch = 6
train_conv_data = DataLoader((data=X_train_conv, label=y_train_conv), batchsize=batch, shuffle=true)
test_conv_data = DataLoader((data=X_test_conv, label=y_test_conv), batchsize=batch, shuffle=true)

# Define the model and move it to GPU
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
) |> gpu

# Loss and Optimizer
loss(x, y) = logitcrossentropy(model(x), y)
optimizer = ADAM(0.001)

# Training loop
epochs = 2
train_epoch_losses = Float64[]
test_epoch_losses = Float64[]
train_epoch_accuracies = Float64[]
test_epoch_accuracies = Float64[]

for epoch in 1:epochs
    println("Epoch: $epoch")
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for (x, y) in train_conv_data
        x, y = x |> gpu, y |> gpu  # Move mini-batch to GPU
        grads = gradient(params(model)) do
            l = loss(x, y)
            train_loss += l
            return l
        end

        update!(optimizer, params(model), grads)

        pred = onecold(model(x)) .- 1  # Adjusting for 0-based indexing
        train_correct += sum(pred .== onecold(y) .- 1)
        train_total += size(y, 2)
    end

    avg_train_loss = train_loss / length(train_conv_data)
    train_accuracy = train_correct / train_total

    push!(train_epoch_losses, avg_train_loss)
    push!(train_epoch_accuracies, train_accuracy)

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    for (x, y) in test_conv_data
        x, y = x |> gpu, y |> gpu  # Move mini-batch to GPU
        l = loss(x, y)
        test_loss += l

        pred = onecold(model(x)) .- 1  # Adjusting for 0-based indexing
        test_correct += sum(pred .== onecold(y) .- 1)
        test_total += size(y, 2)
    end

    avg_test_loss = test_loss / length(test_conv_data)
    test_accuracy = test_correct / test_total

    push!(test_epoch_losses, avg_test_loss)
    push!(test_epoch_accuracies, test_accuracy)

    println("Train Loss: $avg_train_loss, Train Accuracy: $train_accuracy")
    println("Test Loss: $avg_test_loss, Test Accuracy: $test_accuracy")
end
