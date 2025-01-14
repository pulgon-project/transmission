import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from ipdb import  set_trace

color = [value for key, value in mcolors.XKCD_COLORS.items()]

# 使用 'tab10' 调色板
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
set_trace()
# 示例
for i in range(10):
    plt.plot([0, 1, 2], [i, i+1, i+2], label=f'Line {i}')
plt.legend()
plt.show()



