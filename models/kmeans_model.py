from sklearn.cluster import KMeans

class KMeansModel:
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        """
        训练KMeans模型
        :param X: 特征矩阵
        :param y: 标签（KMeans是无监督学习，y可忽略）
        """
        self.model.fit(X)

    def predict(self, X):
        """
        使用训练好的KMeans模型进行预测
        :param X: 特征矩阵
        :return: 预测的聚类标签
        """
        return self.model.predict(X)