import numpy as np  # 导入numpy库
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
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 安装gensim库
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
except ImportError:
    logging.info("gensim库未安装，正在安装...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "gensim==4.2.0"], capture_output=True, text=True)  # 指定兼容的gensim版本
    if result.returncode != 0:
        logging.error("安装gensim库失败，错误信息如下：")
        logging.error(result.stderr)
        sys.exit(1)
    else:
        logging.info("gensim库安装成功")
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
        # 将结果转换为一维数组
        return sum(vectors) / len(vectors)
    else:
        # 返回一个一维零向量
        return [0] * 100

def train_and_evaluate_model(df, y, preprocessor, rf_pipeline, xgb_pipeline, rf_param_grid, xgb_param_grid):
    # 使用GridSearchCV进行随机森林的超参数调优
    rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

    # 使用GridSearchCV进行XGBoost的超参数调优
    xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

    logging.info(f"Shape of X (df[['评论内容']]): {df[['评论内容']].shape}")
    logging.info(f"Length of y: {len(y)}")

    # 将X转换为字符串列表
    X = df['评论内容'].tolist()
    X = np.array(X)  # 将X转换为NumPy数组以便正确索引

    # 训练随机森林模型
    rf_best_model = rf_grid_search.fit(X, y)

    # 训练XGBoost模型
    xgb_best_model = xgb_grid_search.fit(X, y)

    # 输出最佳参数
    logging.info(f"Random Forest Best Parameters: {rf_grid_search.best_params_}")
    logging.info(f"XGBoost Best Parameters: {xgb_grid_search.best_params_}")

    # 使用k折交叉验证评估随机森林模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 设置k=5
    rf_fold_results = []

    for train_index, test_index in kf.split(df):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # 训练模型
        rf_best_model.fit(X_train, y_train)

        # 评估模型
        results = evaluate_model(rf_best_model, X_test, y_test)
        rf_fold_results.append(results)

    # 计算随机森林的平均评估结果
    rf_avg_results = {
        "accuracy": sum(result["accuracy"] for result in rf_fold_results) / len(rf_fold_results),
        "precision": sum(result["precision"] for result in rf_fold_results) / len(rf_fold_results),
        "recall": sum(result["recall"] for result in rf_fold_results) / len(rf_fold_results),
        "f1_score": sum(result["f1_score"] for result in rf_fold_results) / len(rf_fold_results)
    }

    logging.info(f"Random Forest Average Results over 5 Folds: {rf_avg_results}")

    # 使用k折交叉验证评估XGBoost模型
    xgb_fold_results = []

    for train_index, test_index in kf.split(df):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # 训练模型
        xgb_best_model.fit(X_train, y_train)

        # 评估模型
        results = evaluate_model(xgb_best_model, X_test, y_test)
        xgb_fold_results.append(results)

    # 计算XGBoost的平均评估结果
    xgb_avg_results = {
        "accuracy": sum(result["accuracy"] for result in xgb_fold_results) / len(xgb_fold_results),
        "precision": sum(result["precision"] for result in xgb_fold_results) / len(xgb_fold_results),
        "recall": sum(result["recall"] for result in xgb_fold_results) / len(xgb_fold_results),
        "f1_score": sum(result["f1_score"] for result in xgb_fold_results) / len(xgb_fold_results)
    }

    logging.info(f"XGBoost Average Results over 5 Folds: {xgb_avg_results}")

    # 比较两个模型的平均F1分数，选择表现更好的模型
    if rf_avg_results["f1_score"] > xgb_avg_results["f1_score"]:
        best_model = rf_best_model
        best_avg_results = rf_avg_results
    else:
        best_model = xgb_best_model
        best_avg_results = xgb_avg_results

    # 可视化评估结果
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    rf_values = [rf_avg_results[metric] for metric in metrics]
    xgb_values = [xgb_avg_results[metric] for metric in metrics]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=rf_values, label='Random Forest', color='b')
    sns.barplot(x=metrics, y=xgb_values, label='XGBoost', color='g', bottom=rf_values)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    return best_model, best_avg_results

# 定义文本预处理器
def text_preprocessor(text):
    return simple_preprocess(text)

preprocessor = FunctionTransformer(text_preprocessor, validate=False)

# 定义随机森林管道
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('rf', RandomForestClassifier())
])

# 定义XGBoost管道
xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('xgb', xgb.XGBClassifier(eval_metric='logloss'))
])

# 定义随机森林超参数网格
rf_param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

# 定义XGBoost超参数网格
xgb_param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6, 10],
    'xgb__learning_rate': [0.01, 0.1, 0.2]
}

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
    fraudulent_comments = set(fraudulent_df['评论内容'].tolist())

    # 生成标签：正常评论为1，刷单评论为0
    y = [1 if comment not in fraudulent_comments else 0 for comment in comments]

    # 确保y的长度与df的长度一致
    assert len(y) == len(df), "y的长度与df的长度不一致"

    # 检查数据集的维度
    logging.info(f"Feature matrix shape: {df[['评论内容']].shape}")
    logging.info(f"Target vector length: {len(y)}")

    # 确保索引一致
    assert df.index.equals(pd.Index(range(len(df)))), "DataFrame索引不正确"

    # 训练并评估模型
    best_model, best_avg_results = train_and_evaluate_model(df, y, preprocessor, rf_pipeline, xgb_pipeline, rf_param_grid, xgb_param_grid)

    # 持久化模型
    joblib.dump(best_model, 'best_model.pkl')
    logging.info("模型已持久化")

    # 使用最佳模型进行预测
    predictions = best_model.predict(df['评论内容'].tolist())
    fraudulent_indices = [i for i, pred in enumerate(predictions) if pred == 0]
    fraudulent_comments = [comments[i] for i in fraudulent_indices]

    logging.info(f"Best Model Average Results: {best_avg_results}")

if __name__ == "__main__":
    main()