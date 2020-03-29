from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
def naviebayes():
    # 1.加载新闻数据
    # 使用这种方法会从网上下载，默认放的位置是C:\Users\Administrator\
    # 下载成功之后加载会快
    news = fetch_20newsgroups(subset = 'all')
    # 2.数据划分测试集和训练集
    x_train, x_test, y_trian, y_test = \
        train_test_split(news.data, news.target, test_size=0.25)

    # 3.特征提取，按照训练集的特征词重要性提取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    # 按照训练集的分类，统计一下测试集中的词的重要性
    print(tf.get_feature_names())
    x_test = tf.transform(x_test)

    # 4.朴素贝叶斯预测
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_trian)
    y_predict = mlt.predict(x_test)
    print("预测的文章类别为：", y_predict)
    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))



def main():
    naviebayes()

if __name__ == '__main__':
    main()

