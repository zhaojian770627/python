import copy, numpy as np

np.random.seed(0)


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


int2binary = {}  # 整数道其二进制表示的映射
binary_dim = 8
# 计算0 - 256 的二进制表示
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

# 参数设置
alpha = 0.9  # 学习速率
input_dim = 2  # 输入的维度是2，减数和被减数
hidden_dim = 16
output_dim = 1  # 输出维度为1

# 初始化网络
synapse_0 = (2 * np.random.random((input_dim, hidden_dim)) - 1) * .05  # 维度为2*16,2是输入维度，16是隐藏层维度
synapse_1 = (2 * np.random.random((hidden_dim, output_dim)) - 1) * .05
synapse_h = (2 * np.random.random((hidden_dim, hidden_dim)) - 1) * .05

# 用于存放反向传播的权重更新值
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

print(len(int2binary))
