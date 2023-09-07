using Flux,  Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle,params,flatten, DataLoader
using Base.Iterators: repeated, partition
using MLDatasets
using CUDA  # Optional, for GPU support
using Plots
# Load FashionMNIST data
trainset= FashionMNIST(split=:train)
testset = FashionMNIST(split=:test)

train_x,train_y= trainset[:]
test_x,test_y= testset[:]
# Reshape data for CNN
train_x = reshape(train_x, 28, 28, 1, :)
test_x = reshape(test_x, 28, 28, 1, :)

# One-hot encode the labels
train_y = onehotbatch(train_y, 0:9)
test_y = onehotbatch(test_y, 0:9)

# Define the CNN
model = Chain(
    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, relu),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, relu),
    MaxPool((2,2)),
    flatten,
    Dense(32, 10),
    Dropout(0.25),
    softmax
)

# Uncomment to run the model on GPU
# train_x, train_y, test_x, test_y, model = train_x |> gpu, train_y |> gpu, test_x |> gpu, test_y |> gpu, model |> gpu

# Loss function
loss(x, y) = crossentropy(model(x), y)

# Optimizer
opt = ADAM()
losses_train = []
losses_test = []
accuracy_train = []
accuracy_test = []

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
function lr_schedule(epoch)
    if epoch < 10
        return 0.001
    elseif epoch < 20
        return 0.0005
    else
        return 0.0001
    end
end

function add_noise(x)
    noise_factor = oftype(x[1],0.5)
    x = x .+ noise_factor * rand32(size(x)...)
    return clamp.(x, oftype(x[1],0), oftype(x[1],))
end


# Training data
batch_size = 128
train_data = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)


# Training loop
epochs = 30
for epoch in 1:epochs
    for (x, y) in train_data
        global opt
        opt = Flux.Optimiser(ADAM(lr_schedule(epoch)), WeightDecay(0.0005))
        gs = Flux.gradient(params(model)) do
            loss(x, y)
        end
        Flux.update!(opt, params(model), gs)
    end
    println("Epoch: $epoch")
    push!(losses_train, loss(train_x, train_y))
    push!(losses_test, loss(test_x, test_y))
    push!(accuracy_train, accuracy(train_x, train_y))
    push!(accuracy_test, accuracy(test_x, test_y))
end

p1 = plot(1:epochs, accuracy_train, label="Train Accuracy", linewidth=2)
plot!(1:epochs, accuracy_test, label="Test Accuracy", linewidth=2)
p2 = plot(1:epochs, losses_train, label="Test Loss", linewidth=2)
plot!(1:epochs, losses_test, label="Test Loss", linewidth=2)
plot(p1, p2, layout=(1, 2), legend=:bottomright, xlabel="Epochs")
# Calculate accuracy
# Evaluate the model
accuracy(train_x, train_y)
accuracy(test_x, test_y)

model(test_x)
test_y