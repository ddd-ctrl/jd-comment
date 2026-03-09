import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import random
import jieba
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def text_augmentation(text, augmentation_rate=0.3):
    """中文文本数据增强：随机删除和随机交换词语"""
    if not text:
        return text
    
    words = list(jieba.cut(text))
    
    if len(words) == 0:
        return text
    
    augmented_words = words.copy()
    
    # 随机删除 - 只删除非关键词语（长度>1 的词更可能是关键词）
    if random.random() < augmentation_rate and len(augmented_words) > 2:
        deletable_indices = [i for i in range(len(augmented_words)) if len(augmented_words[i]) == 1]
        if deletable_indices:
            delete_idx = random.choice(deletable_indices)
            augmented_words.pop(delete_idx)
    
    # 随机交换 - 只交换相邻或相近的词语以保持语义
    if len(augmented_words) > 2 and random.random() < augmentation_rate:
        idx1 = random.randint(0, len(augmented_words) - 1)
        # 只与相邻或间隔 1 个位置的词交换
        idx2_range = [i for i in range(max(0, idx1-2), min(len(augmented_words), idx1+3)) if i != idx1]
        if idx2_range:
            idx2 = random.choice(idx2_range)
            augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]
    
    return ''.join(augmented_words)

def evaluate_model(model, X, y):
    """评估模型性能"""
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average='weighted', zero_division=0))
    }

def augment_dataset(df, y, target_class=0, augmentation_factor=2):
    """对少数类进行数据增强
    
    注意：数据增强应该只在训练集上进行，不应该在验证集/测试集上进行
    此函数仅用于增加训练数据的多样性，避免过拟合
    """
    y = np.array(y)
    df = df.copy()
    df['标签'] = y
    
    # 分离少数类和多数类
    minority_indices = df[df['标签'] == target_class].index.tolist()
    majority_indices = df[df['标签'] != target_class].index.tolist()
    
    logging.info(f"原始数据集 - 少数类：{len(minority_indices)}, 多数类：{len(majority_indices)}")
    
    if len(minority_indices) == 0:
        logging.warning("没有找到少数类样本，跳过数据增强")
        return df.drop('标签', axis=1), y
    
    # 只对少数类进行增强
    augmented_texts = []
    augmented_labels = []
    
    minority_texts = df.loc[minority_indices, '评论内容'].tolist()
    for text in minority_texts:
        for _ in range(augmentation_factor):
            augmented_text = text_augmentation(text)
            # 确保增强后的文本与原文本不同
            if augmented_text != text:
                augmented_texts.append(augmented_text)
                augmented_labels.append(target_class)
    
    logging.info(f"生成了 {len(augmented_texts)} 条增强样本")
    
    # 创建增强数据的 DataFrame
    if augmented_texts:
        augmented_df = pd.DataFrame({
            '评论内容': augmented_texts,
            '标签': augmented_labels
        })
        
        # 只合并多数类和增强后的少数类（不重复包含原始少数类）
        majority_df = df.loc[majority_indices, ['评论内容', '标签']]
        combined_df = pd.concat([majority_df, augmented_df], ignore_index=True)
    else:
        combined_df = df[['评论内容', '标签']]
    
    combined_y = combined_df['标签'].tolist()
    
    logging.info(f"增强后数据集 - 总样本数：{len(combined_df)}")
    
    return combined_df.drop('标签', axis=1), combined_y

def train_and_evaluate_model(df, y, rf_pipeline, xgb_pipeline, use_augmentation=False):
    """训练并评估模型
    
    Args:
        use_augmentation: 是否在训练集上应用数据增强（只在训练集上，不在测试集上）
    """
    rf_simple_param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [3, 5, 10],
        'rf__min_samples_split': [3, 5]
    }
    
    xgb_simple_param_grid = {
        'xgb__n_estimators': [50, 100],
        'xgb__max_depth': [2, 4, 6],
        'xgb__learning_rate': [0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    X = np.array(df['评论内容'].tolist())
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rf_fold_results = []
    xgb_fold_results = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        logging.info(f"正在处理第 {fold + 1}/3 折...")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        
        # 关键修复：只在训练集上进行数据增强，避免数据泄露
        if use_augmentation:
            train_df_fold = pd.DataFrame({'评论内容': X_train, '标签': y_train})
            train_df_augmented, y_train_augmented = augment_dataset(
                train_df_fold, y_train, target_class=0, augmentation_factor=2
            )
            X_train_aug = np.array(train_df_augmented['评论内容'].tolist())
            y_train = y_train_augmented
            X_train = X_train_aug
            logging.info(f"第 {fold + 1} 折增强后训练集大小：{len(X_train)}")
        
        # 使用增强后的训练集进行网格搜索
        rf_grid_search = GridSearchCV(rf_pipeline, rf_simple_param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_simple_param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        
        rf_grid_search.fit(X_train, y_train)
        xgb_grid_search.fit(X_train, y_train)
        
        if fold == 0:
            logging.info(f"Random Forest Best Parameters: {rf_grid_search.best_params_}")
            logging.info(f"XGBoost Best Parameters: {xgb_grid_search.best_params_}")
        
        rf_fold_results.append(evaluate_model(rf_grid_search.best_estimator_, X_test, y_test))
        xgb_fold_results.append(evaluate_model(xgb_grid_search.best_estimator_, X_test, y_test))
    
    rf_avg_results = {
        "accuracy": sum(r["accuracy"] for r in rf_fold_results) / len(rf_fold_results),
        "precision": sum(r["precision"] for r in rf_fold_results) / len(rf_fold_results),
        "recall": sum(r["recall"] for r in rf_fold_results) / len(rf_fold_results),
        "f1_score": sum(r["f1_score"] for r in rf_fold_results) / len(rf_fold_results)
    }
    
    xgb_avg_results = {
        "accuracy": sum(r["accuracy"] for r in xgb_fold_results) / len(xgb_fold_results),
        "precision": sum(r["precision"] for r in xgb_fold_results) / len(xgb_fold_results),
        "recall": sum(r["recall"] for r in xgb_fold_results) / len(xgb_fold_results),
        "f1_score": sum(r["f1_score"] for r in xgb_fold_results) / len(xgb_fold_results)
    }
    
    logging.info(f"Random Forest Average Results: {rf_avg_results}")
    logging.info(f"XGBoost Average Results: {xgb_avg_results}")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    rf_values = [rf_avg_results[m] for m in metrics]
    xgb_values = [xgb_avg_results[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=rf_values, label='Random Forest', color='b')
    sns.barplot(x=metrics, y=xgb_values, label='XGBoost', color='g', bottom=rf_values)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    
    if rf_avg_results["f1_score"] > xgb_avg_results["f1_score"]:
        best_model = rf_grid_search.best_estimator_
    else:
        best_model = xgb_grid_search.best_estimator_
    
    return best_model, rf_avg_results if rf_avg_results["f1_score"] > xgb_avg_results["f1_score"] else xgb_avg_results

def main():
    try:
        df = pd.read_csv("data/comments.csv")
        fraudulent_df = pd.read_csv("data/fraudulent_comments.csv")
    except FileNotFoundError as e:
        logging.error(f"文件未找到：{e}")
        return
    
    comments = df['评论内容'].tolist()
    fraudulent_comments_set = set(fraudulent_df['评论内容'].tolist())
    y = [1 if comment not in fraudulent_comments_set else 0 for comment in comments]
    
    assert len(y) == len(df), "y 的长度与 df 的长度不一致"
    
    logging.info(f"原始数据集大小：{len(df)} 条评论")
    logging.info(f"正常评论：{sum(y)}")
    logging.info(f"刷单评论：{len(y) - sum(y)}")
    
    # 数据增强将在交叉验证的每个训练折内部进行，避免数据泄露
    use_augmentation = len(df) < 1000
    if use_augmentation:
        logging.info("检测到小数据集，将在交叉验证的训练集上应用数据增强技术...")
    
    tfidf_params = {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.9,
        'ngram_range': (1, 2)
    }
    
    rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    xgb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    
    y_np = np.array(y).astype(int)
    unique_classes = np.unique(y_np)
    
    if len(unique_classes) > 0:
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_np)
        class_weights_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}
        rf_pipeline.named_steps['rf'].class_weight = class_weights_dict
        
        if len(unique_classes) == 2 and 0 in unique_classes and 1 in unique_classes:
            count_0 = np.sum(y_np == 0)
            count_1 = np.sum(y_np == 1)
            xgb_pipeline.named_steps['xgb'].scale_pos_weight = count_1 / count_0
        else:
            xgb_pipeline.named_steps['xgb'].scale_pos_weight = 1
    
    best_model, best_avg_results = train_and_evaluate_model(
        df, y, rf_pipeline, xgb_pipeline, use_augmentation=use_augmentation
    )
    
    joblib.dump(best_model, 'models_output/best_model.pkl')
    logging.info("模型已持久化到 models_output/best_model.pkl")
    
    predictions = best_model.predict(df['评论内容'].tolist())
    fraudulent_indices = [i for i, pred in enumerate(predictions) if pred == 0]
    detected_fraudulent = [df['评论内容'].iloc[i] for i in fraudulent_indices]
    
    logging.info(f"最佳模型平均结果：{best_avg_results}")
    logging.info(f"识别出的刷单评论数量：{len(detected_fraudulent)}")
    
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
