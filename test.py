import numpy as np

# 定义投影矩阵，这里假设已经有了每个子空间的投影矩阵
# 假设投影矩阵分别为 P1, P2, ..., P6
# 这里假设投影矩阵已经是复数形式
P1 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)
P2 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)
P3 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)
P4 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)
P5 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)
P6 = np.random.randn(18, 108) + 1j * np.random.randn(18, 108)

# 生成随机的复数向量数组 array1，形状为 (5, 108)
array1 = np.random.randn(5, 108) + 1j * np.random.randn(5, 108)

# 定义合并子空间的函数
def merge_subspaces(projections):
    # 合并每个子空间的投影向量
    merged_vector = np.sum(projections, axis=0)
    # 归一化合并后的向量
    merged_vector /= np.linalg.norm(merged_vector)
    return merged_vector

# 定义重新正交化函数
def reorthogonalize(vectors):
    # 对向量集进行 Gram-Schmidt 正交化
    Q, _ = np.linalg.qr(vectors.T, mode='complete')
    return Q.T

# 将 array1 中的每个向量投影到各自的子空间中
projections = [np.dot(P1, array1[i]) for i in range(5)]
projections += [np.dot(P2, array1[i]) for i in range(5)]
projections += [np.dot(P3, array1[i]) for i in range(5)]
projections += [np.dot(P4, array1[i]) for i in range(5)]
projections += [np.dot(P5, array1[i]) for i in range(5)]
projections += [np.dot(P6, array1[i]) for i in range(5)]

# 合并投影向量并归一化
merged_vector = merge_subspaces(projections)

# 重新正交化
array2 = reorthogonalize(np.array([merged_vector]))

# 打印结果
print("array2 shape:", array2.shape)
