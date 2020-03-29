from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def tf_idfvec():
    """
    文本特征值提取
    :rtype: None
    """
    tfidf = TfidfVectorizer()
    str = ["""今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。""",
           """我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。""",
           """如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"""]
    # 中文分词
    cut_str = chinese_cut(str)
    data = tfidf.fit_transform(cut_str)
    print(tfidf.get_feature_names())
    print(data.toarray())
    return None

def chinese_cut(content):
    """
    处理汉字分词
    :param content: 需要处理的文章列表
    :return: 处理后的文章列表
    """
    # 返回的处理后的文章列表
    res = list()
    # 遍历处理列表中的每一篇文章
    for article in content:
        # jieba 分词
        con_tmp = jieba.cut(article)
        # con 是生成器，里面是一个个的词，需要转换成列表形式
        con = list(con_tmp)
        # 把单个词用空格拼接成字符串
        str = " ".join(con)
        # 放入返回列表
        res.append(str)
    return res
def main():
    tf_idfvec()

if __name__ == "__main__":
    main()