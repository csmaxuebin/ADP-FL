import matplotlib.pyplot as plt
import numpy as np

# 自定义横坐标值和纵坐标范围
# ======================dp2====================================
custom_x = ['3', '5', '7']
values1 = [42.55, 42.84, 43.56]
values2 = [41.66, 41.78, 42.38]

# ======================dp5====================================
# 创建x坐标轴位置
x = np.arange(len(custom_x))

# 设置纵坐标范围
plt.ylim(0, 60)

# 创建三个柱状图，使用不同的颜色
plt.bar(x-0.1, values1, 0.2, label='ADP-FL', color='None', hatch='//', edgecolor='#73BCD5')
plt.bar(x+0.1, values2, 0.2, label='ADP-FL (Fixed ε)', color='None', hatch='\\\\', edgecolor='#FFD06E')

# 设置横坐标刻度位置和标签
plt.xticks(x, custom_x)

# 添加标签和图例
plt.xlabel('privacy budget ε')
plt.ylabel('Accuracy')

plt.legend()

# 在每个柱状图上方添加数字标签
for i, v in enumerate(values1):
    plt.text(i - 0.1, v + 0.1, str(v), color='black', ha='center')

for i, v in enumerate(values2):
    plt.text(i+0.1, v + 0.1, str(v), color='black', ha='center')

# 自动调整布局，避免标签重叠
plt.tight_layout()

# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300,format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()
