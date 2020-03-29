from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 数据集获取 实例化，返回的是字典类型
li = load_iris()
# 特征值
# print("获取特征值：\n", li.data)
# 目标值
# print("获取目标值：\n", li.target_names)

# 下面划分数据集
# 注意：这里的 目标值 指的就是 标签值
# 参数分别为数据集的特征值，数据集的标签值，test_size为测试集的大小
# 返回值，包含训练集特征值、测试集特征值、训练集目标值、测试集目标值
# 这个函数只是起到一个分割数据的功能，这里面的数据化都是之前获取的
# 分割之后的数据是乱序的
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)