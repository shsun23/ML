from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def var_th():
    """
    特征选择，删除低方差的特征
    :return: None
    """
    # 实例化
    var_th = VarianceThreshold(threshold=0.0)
    # 调用
    data = var_th.fit_transform([[0, 2, 0, 3],
                                [0, 1, 4, 3],
                                [0, 1, 1, 3]])
    print(data)


def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],
                                [6,3,0,8],
                                [5,4,9,1]])
    print(data)
def main():
    # var_th()
    pca()


if __name__ == '__main__':
    main()