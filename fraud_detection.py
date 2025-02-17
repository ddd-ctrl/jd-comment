import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def detect_fraudulent_comments(file_path):
    """
    识别刷单行为
    :param file_path: 评论数据文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 提取评论内容
    comments = df['评论内容'].tolist()

    # 使用TF-IDF提取文本特征
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
    X = vectorizer.fit_transform(comments)

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # 查看聚类结果
    for cluster in df['cluster'].unique():
        print(f"Cluster {cluster}:")
        print(df[df['cluster'] == cluster]['评论内容'].head(2))
        print("\n")

    # 识别刷单行为
    fraudulent_comments = df[df['cluster'] == 0]  # 假设聚类结果为0的评论为刷单评论
    print("可疑的刷单评论：")
    print(fraudulent_comments[['用户', '评论内容', '评论时间']])

    # 将可疑刷单评论保存到文件
    fraudulent_comments.to_csv("d:\\jd-comment\\fraudulent_comments.csv", index=False, encoding='utf-8')
    print("可疑刷单评论已保存到 fraudulent_comments.csv")

if __name__ == "__main__":
    # 独立运行刷单识别功能
    detect_fraudulent_comments("d:\\jd-comment\\comments.csv")