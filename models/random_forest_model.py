from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        """
        训练随机森林模型
        :param X: 特征矩阵
        :param y: 标签
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        使用训练好的随机森林模型进行预测
        :param X: 特征矩阵
        :return: 预测的标签
        """
        return self.model.predict(X)