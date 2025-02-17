import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from models.kmeans_model import KMeansModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel

def evaluate_model(model, X, y):
    """
    评估模型性能
    :param model: 机器学习模型
    :param X: 特征矩阵
    :param y: 真实标签
    :return: 评估结果字典
    """
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y, y_pred, average='weighted', zero_division=0)
    }

def main():
    # 读取评论数据
    df = pd.read_csv("d:\\jd-comment\\comments.csv")
    comments = df['评论内容'].tolist()

    # 使用TF-IDF提取文本特征
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
    X = vectorizer.fit_transform(comments)

    # 读取刷单评论数据
    fraudulent_df = pd.read_csv("d:\\jd-comment\\fraudulent_comments.csv")
    fraudulent_comments = fraudulent_df['评论内容'].tolist()

    # 生成标签：正常评论为1，刷单评论为0
    y = [1 if comment not in fraudulent_comments else 0 for comment in comments]

    # 初始化并训练不同模型
    models = {
        "KMeans": KMeansModel(),
        "SVM": SVMModel(),
        "RandomForest": RandomForestModel()
    }

    # 评估并输出结果
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        model.fit(X, y)
        results = evaluate_model(model, X, y)
        print(f"{model_name} Results: {results}")

if __name__ == "__main__":
    main()