import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
# resnet34 + cifar100
values1 = [44.59, 44.59, 44.59, 44.59, 44.59]
values2 = [41.74, 42.55, 42.84, 43.56, 43.87]
values3 = [21.56, 24.76, 27.53, 28.28, 28.37]
values4 = [1.41, 10.4, 12.54, 15.89, 21.76]

# 设置纵坐标范围为0到100
plt.ylim(0, 50)

# 使用自定义的x轴坐标点
rounds = [1, 3, 5, 7, 10]

# 绘制每个文件的准确率数据
plt.plot(rounds, values1, color='#ef8a46', marker='*', label='NO-DP')
plt.plot(rounds, values2, color='#FFD06E', marker='p', label='ADP-FL')
plt.plot(rounds, values3, color='#73BCD5', marker='s', label='ADP-FedPer')
plt.plot(rounds, values4, color='#386795', marker='o', label='ADP-FedAvg')

# 设置x轴刻度
# 设置x轴刻度
plt.xticks(range(1, 11))

# 添加标签和图例
plt.xlabel('privacy budget ε')
plt.ylabel('Accuracy')
plt.legend()
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)  # Adjust the bbox_to_anchor and ncol as needed
# plt.subplots_adjust(bottom=0.2)
# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300, format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()
