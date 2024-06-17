import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# 隐私预算为3时，our 谷歌 固定精度对比res18+cifar10
# 加载数据
with open('plant_data/2.1-our-dp3-75.99.txt', 'r') as f1:
    data1 = f1.readlines()  # 加载第一个文件的数据

with open('plant_data/2.1-gg-dp3-71.94.txt', 'r') as f2:
    data2 = f2.readlines()  # 加载第二个文件的数据

with open('plant_data/2.1-gd4-dp3-72.97.txt', 'r') as f3:
    data3 = f3.readlines()  # 加载第三个文件的数据

with open('plant_data/2.1-gd0.5-dp3-72.79.txt', 'r') as f4:
    data4 = f4.readlines()  # 加载第四个文件的数据
#
with open('plant_data/2.1-zxj-dp3-74.txt', 'r') as f5:
    data5 = f5.readlines()  # 加载第五个文件的数据

# 从每行数据中提取准确率值，分别保存到两个数组中
acc1 = [float(line.split()[-1]) for line in data1][:100]
acc2 = [float(line.split()[-1]) for line in data2][:100]
acc3 = [float(line.split()[-1]) for line in data3][:100]
acc4 = [float(line.split()[-1]) for line in data4][:100]
acc5 = [float(line.split()[-1]) for line in data5][:100]

# 使用步长为10的range函数生成x轴坐标
rounds = list(range(1, 100, 5))

# 设置纵坐标范围为0到100
plt.ylim(40, 80)
y_ticks = [i * 10 for i in range(11)]
y_major_locator = FixedLocator(y_ticks)
plt.gca().yaxis.set_major_locator(y_major_locator)

# 绘制每个文件的准确率数据
plt.plot(rounds, acc1[::5], color='#ef8a46', marker='*', label='ADP-FL')
plt.plot(rounds, acc2[::5], color='#E86254', marker='o', label='Adaptive')
plt.plot(rounds, acc3[::5], color='#73BCD5', marker='s', label='Fixed (C=4)')
plt.plot(rounds, acc4[::5], color='#386795', marker='p', label='Fixed (C=0.5)')
plt.plot(rounds, acc5[::5], color='#FFD06E', marker='^', label='ULDP-FED')

# 绘制每个文件的准确率数据
# plt.plot(rounds, acc1, color='darkorange',  label='FedAvg')
# plt.plot(rounds, acc2, color='springgreen',  label='FedProx')
# plt.plot(rounds, acc3, color='gold', label='FedPer')
# plt.plot(rounds, acc4, color='red',  label='FedVF')
# plt.plot(rounds, acc5, color='cornflowerblue',  label='PDP-FD')
# plt.plot(rounds, acc6, color='violet',  label='dynamic_Positive>negative_--')

# 添加标签和图例
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.title('Training Results')
plt.legend()

# 保存图为svg格式，即矢量图格式
plt.savefig("picture/acc.svg", dpi=300,format="svg")

# # 保存图为eps格式
plt.savefig("picture/acc.eps", dpi=300, format="eps")

# 显示图表
plt.show()

