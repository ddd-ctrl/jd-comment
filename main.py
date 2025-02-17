import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess
import sys
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess




# 安装gensim库
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
except ImportError:
    print("gensim库未安装，正在安装...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "gensim==4.2.0"], capture_output=True, text=True)  # 指定兼容的gensim版本
    if result.returncode != 0:
        print("安装gensim库失败，错误信息如下：")
        print(result.stderr)
        sys.exit(1)
    else:
        print("gensim库安装成功")
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess

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
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average='weighted', zero_division=0))
    }

def get_comment_vector(comment, model_w2v):
    words = simple_preprocess(comment)
    vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    if vectors:
        # 将结果直接返回为一维数组
        return sum(vectors) / len(vectors)
    else:
        # 返回一个一维零向量
        return [0] * 100

def main():
    # 读取评论数据
    df = pd.read_csv("d:\\jd-comment\\comments.csv")
    comments = df['评论内容'].tolist()

    # 预处理评论数据
    processed_comments = [simple_preprocess(comment) for comment in comments]

    # 训练Word2Vec模型
    model_w2v = Word2Vec(sentences=processed_comments, vector_size=100, window=5, min_count=1, workers=4)

    # 获取评论的Word2Vec向量表示
    X_w2v = [get_comment_vector(comment, model_w2v) for comment in comments]

    # 读取刷单评论数据
    fraudulent_df = pd.read_csv("d:\\jd-comment\\fraudulent_comments.csv")
    fraudulent_comments = fraudulent_df['评论内容'].tolist()

    # 生成标签：正常评论为1，刷单评论为0
    y = [1 if comment not in fraudulent_comments else 0 for comment in comments]

    # 初始化TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    # 创建一个转换器，将评论内容转换为TF-IDF特征
    tfidf_transformer = FunctionTransformer(lambda x: tfidf_vectorizer.fit_transform(x).toarray(), validate=False)

    # 创建一个转换器，将评论内容转换为Word2Vec特征
    w2v_transformer = FunctionTransformer(lambda x: [get_comment_vector(comment, model_w2v) for comment in x], validate=False)

    # 创建一个列转换器，将TF-IDF特征和Word2Vec特征组合在一起
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', tfidf_transformer, '评论内容'),
            ('w2v', w2v_transformer, '评论内容')
        ])

    # 创建一个管道，将预处理器和随机森林分类器组合在一起
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 定义超参数网格
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(df[['评论内容']], y)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 使用k折交叉验证评估模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 设置k=5
    fold_results = []

    for train_index, test_index in kf.split(df):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # 训练模型
        best_model.fit(X_train, y_train)

        # 评估模型
        results = evaluate_model(best_model, X_test, y_test)
        fold_results.append(results)

    # 计算平均评估结果
    avg_results = {
        "accuracy": sum(result["accuracy"] for result in fold_results) / len(fold_results),
        "precision": sum(result["precision"] for result in fold_results) / len(fold_results),
        "recall": sum(result["recall"] for result in fold_results) / len(fold_results),
        "f1_score": sum(result["f1_score"] for result in fold_results) / len(fold_results)
    }

    print(f"Average Results over 5 Folds: {avg_results}")

if __name__ == "__main__":
    main()
