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
print(len(int2binary))
