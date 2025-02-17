from sklearn.svm import SVC

class SVMModel:
    def __init__(self, kernel='linear', random_state=42):
        self.model = SVC(kernel=kernel, random_state=random_state)

    def fit(self, X, y):
        """
        训练SVM模型
        :param X: 特征矩阵
        :param y: 标签
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        使用训练好的SVM模型进行预测
        :param X: 特征矩阵
        :return: 预测的标签
        """
        return self.model.predict(X)