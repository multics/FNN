using Flux

ones_initializer = (dims...) -> ones(Float32, dims...)

# 定义前馈神经网络（FNN）
struct FNN
    model::Chain
end

function FNN(input_dim::Int, hidden_dim::Int, output_dim::Int)
    model = Chain(
        Dense(input_dim => hidden_dim, relu; bias=ones(Float32, hidden_dim), init=ones_initializer),  # 隐藏层
        Dense(hidden_dim => output_dim; bias=ones(Float32, output_dim), init=ones_initializer)        # 输出层
    )
    return FNN(model)
end

# 前向传播函数
function predict(fn::FNN, x::AbstractArray)
    return fn.model(x)
end

# 训练函数
function train!(fn::FNN, x_train, y_train; epochs=100, lr=0.01)
    loss_fn(y_pred, y_true) = Flux.Losses.mse(y_pred, y_true)
    opt = Flux.Optimiser(Flux.Descent(lr))  # 下降优化器
    ps = Flux.params(fn.model)

    for epoch in 1:epochs
        loss, grads = Flux.withgradient(ps) do
            loss_fn(fn.model(x_train), y_train)
        end
        Flux.update!(opt, ps, grads)

        if epoch % 10 == 0
            println("Epoch $epoch, Loss: $loss")
        end
    end
end

# 生成测试数据
x_train = rand(2, 100)   # 输入特征 (2维，100个样本)
y_train = rand(1, 100)   # 输出 (1维)

input = 2

# 创建 FNN 实例
fnn = FNN(input, 2, 1)

# 测试前向传播函数
x_test = ones(Float32, input, 1)
# x_test = zeros(Float32, 1)
# x_test = Float32.([1.0; 0.0; 0.0; 0.0; 0.0])
y_pred = predict(fnn, x_test)

# # 训练网络
# train!(fnn, x_train, y_train, epochs=100, lr=0.01)

# # 测试网络
# x_test = rand(2, 5)  # 5 个新的输入样本
# y_pred = predict(fnn, x_test)

println("输入特征: ", x_test)
println("\nmodel: ", fnn.model)
for layer in fnn.model.layers
    println("\nLayer: ", layer)
    if layer isa Dense
        println("Weights: ", layer.weight)
        println("Bias: ", layer.bias)
    end
end
println("\n预测结果: ", y_pred)
