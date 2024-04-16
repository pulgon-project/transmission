import numpy as np
import matplotlib.pyplot as plt

# 创建一个 5x5 的随机矩阵
matrix = np.random.rand(5, 5)

# 绘制矩阵的热图
plt.matshow(matrix, cmap='viridis')  # cmap参数指定了颜色映射，这里使用了viridis颜色映射
plt.colorbar()  # 添加颜色条

# 更改 x 轴刻度和标签
x_values = [3, 2, 1, 5, 6]
plt.xticks(np.arange(len(x_values)), x_values)
plt.yticks(np.arange(len(x_values)), x_values)

# 在 x 轴和 y 轴上添加数字
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')

plt.show()
