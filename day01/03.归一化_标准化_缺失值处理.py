from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 缺失值处理
from sklearn.impute import SimpleImputer
import numpy as np

def mm():
    """
    归一化
    :return: None
    """
    # 1.实例化
    mm = MinMaxScaler(feature_range=(0,2))
    # 2.归一化
    data = mm.fit_transform([[90,2,10,40],
                        [60,4,15,45],
                        [75,3,13,46]])
    print(data)


def stand():
    """
    标准化
    :return: None
    """
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],
                        [ 2., 4., 2.],
                        [ 4., 6., -1.]])
    print(data)

def im():
    """
    缺失值处理
    :return: None
    """
    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = im.fit_transform([[1, 2],
                    [np.nan, 3],
                    [7, 6]])
    print(data)


def main():
    # mm()
    # stand()
    im()

if __name__ == "__main__":
    main()