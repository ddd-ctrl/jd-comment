import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import random
import jieba
import re
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
    
    if random.random() < augmentation_rate and len(augmented_words) > 2:
        deletable_indices = [i for i in range(len(augmented_words)) if len(augmented_words[i]) == 1]
        if deletable_indices:
            delete_idx = random.choice(deletable_indices)
            augmented_words.pop(delete_idx)
    
    if len(augmented_words) > 2 and random.random() < augmentation_rate:
        idx1 = random.randint(0, len(augmented_words) - 1)
        idx2_range = [i for i in range(max(0, idx1-2), min(len(augmented_words), idx1+3)) if i != idx1]
        if idx2_range:
            idx2 = random.choice(idx2_range)
            augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]
    
    return ''.join(augmented_words)

def extract_text_features(texts):
    """提取文本统计特征"""
    features = []
    
    positive_words = {'好', '不错', '优秀', '满意', '喜欢', '棒', '完美', '推荐', '值得', '好评', '正品', '划算', '实惠', '赞', '喜欢'}
    negative_words = {'差', '不好', '失望', '讨厌', '垃圾', '假货', '坑', '骗', '糟糕', '后悔', '差评', '贵', '坑人', '垃圾'}
    fraud_keywords = {'好评', '推荐', '五星', '满分', '正品', '划算', '便宜', '实惠', '赞', '不错', '很好', '非常好'}
    
    for text in texts:
        words = list(jieba.cut(text))
        words_set = set(words)
        
        # 1. 基础统计特征
        text_length = len(text)
        word_count = len(words)
        avg_word_length = text_length / word_count if word_count > 0 else 0
        
        # 2. 标点符号特征
        punctuation_count = len(re.findall(r'[,.!?.,,.!?]', text))
        exclamation_count = text.count('!') + text.count('！')
        question_count = text.count('?') + text.count('?')
        
        # 3. 情感词特征
        positive_count = len(words_set & positive_words)
        negative_count = len(words_set & negative_words)
        sentiment_ratio = positive_count / (negative_count + 1)
        
        # 4. 刷单关键词特征
        fraud_keyword_count = len(words_set & fraud_keywords)
        has_fraud_keyword = 1 if fraud_keyword_count > 0 else 0
        
        # 5. 词汇多样性
        unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
        
        # 6. 形容词/副词比例（简化版）
        descriptive_words = sum(1 for w in words if len(w) == 1 and w in {'很', '非常', '特别', '太', '真', '超'})
        
        features.append([
            text_length,
            word_count,
            avg_word_length,
            punctuation_count,
            exclamation_count,
            question_count,
            positive_count,
            negative_count,
            sentiment_ratio,
            fraud_keyword_count,
            has_fraud_keyword,
            unique_word_ratio,
            descriptive_words
        ])
    
    return np.array(features)

def augment_dataset(df, y, target_class=0, augmentation_factor=2):
    """对少数类进行数据增强"""
    y = np.array(y)
    df = df.copy()
    df['标签'] = y
    
    minority_indices = df[df['标签'] == target_class].index.tolist()
    majority_indices = df[df['标签'] != target_class].index.tolist()
    
    logging.info(f"原始数据集 - 少数类：{len(minority_indices)}, 多数类：{len(majority_indices)}")
    
    if len(minority_indices) == 0:
        logging.warning("没有找到少数类样本，跳过数据增强")
        return df.drop('标签', axis=1), y
    
    augmented_texts = []
    augmented_labels = []
    
    minority_texts = df.loc[minority_indices, '评论内容'].tolist()
    for text in minority_texts:
        for _ in range(augmentation_factor):
            augmented_text = text_augmentation(text)
            if augmented_text != text:
                augmented_texts.append(augmented_text)
                augmented_labels.append(target_class)
    
    logging.info(f"生成了 {len(augmented_texts)} 条增强样本")
    
    if augmented_texts:
        augmented_df = pd.DataFrame({
            '评论内容': augmented_texts,
            '标签': augmented_labels
        })
        majority_df = df.loc[majority_indices, ['评论内容', '标签']]
        combined_df = pd.concat([majority_df, augmented_df], ignore_index=True)
    else:
        combined_df = df[['评论内容', '标签']]
    
    combined_y = combined_df['标签'].tolist()
    
    logging.info(f"增强后数据集 - 总样本数：{len(combined_df)}")
    
    return combined_df.drop('标签', axis=1), combined_y

def create_enhanced_features(comments, tfidf_vectorizer=None, fit=True):
    """创建增强的特征：TF-IDF + 文本统计特征"""
    # 1. TF-IDF 特征
    if fit:
        tfidf_features = tfidf_vectorizer.fit_transform(comments)
    else:
        tfidf_features = tfidf_vectorizer.transform(comments)
    
    # 2. 文本统计特征
    text_stats = extract_text_features(comments)
    
    # 3. 合并特征
    text_stats_sparse = sparse.csr_matrix(text_stats)
    enhanced_features = sparse.hstack([tfidf_features, text_stats_sparse])
    
    return enhanced_features, tfidf_vectorizer

def apply_smote(X, y):
    """应用 SMOTE 过采样"""
    logging.info("正在应用 SMOTE 过采样...")
    
    # 检查是否可以应用 SMOTE
    min_class_count = min(np.sum(np.array(y) == c) for c in np.unique(y))
    
    if min_class_count < 2:
        logging.warning("少数类样本太少，无法应用 SMOTE，跳过...")
        return X, y
    
    # 确保每个类别至少有 k+1 个样本
    k_neighbors = min(5, min_class_count - 1)
    if k_neighbors < 1:
        logging.warning("少数类样本太少，无法设置 SMOTE 参数，跳过...")
        return X, y
    
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logging.info(f"SMOTE 后样本数：{len(y_resampled)} (原始：{len(y)})")
    
    return X_resampled, y_resampled

def build_ensemble_model():
    """构建集成学习模型"""
    # 1. 基模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    
    # 2. Voting Classifier (软投票)
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_clf),
            ('lgb', lgb_clf),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft',
        weights=[2, 2, 2, 1, 1]
    )
    
    # 3. Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_clf),
            ('lgb', lgb_clf)
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return {
        'voting': voting_clf,
        'stacking': stacking_clf,
        'rf': rf,
        'xgb': xgb_clf,
        'lgb': lgb_clf
    }

def optimize_threshold(y_true, y_pred_proba):
    """优化分类阈值"""
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold

def evaluate_model(y_true, y_pred, y_pred_proba=None, method_name=""):
    """评估模型性能"""
    results = {
        'method': method_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }
    
    # 如果提供了概率，计算 AUC
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score
        results['auc'] = float(roc_auc_score(y_true, y_pred_proba))
    
    return results

def train_and_evaluate_enhanced(df, y, use_augmentation=True, use_smote=True, use_ensemble=True):
    """训练并评估增强模型"""
    logging.info("="*60)
    logging.info("增强模型训练开始")
    logging.info("="*60)
    
    comments = df['评论内容'].tolist()
    y = np.array(y)
    
    # 交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = {
        'voting': [],
        'stacking': [],
        'rf': [],
        'xgb': [],
        'lgb': []
    }
    
    best_models = {}
    optimal_thresholds = {}
    
    for fold, (train_index, test_index) in enumerate(kf.split(comments, y)):
        logging.info(f"\n{'='*60}")
        logging.info(f"第 {fold + 1}/5 折")
        logging.info(f"{'='*60}")
        
        # 划分训练集和测试集
        train_comments = [comments[i] for i in train_index]
        test_comments = [comments[i] for i in test_index]
        y_train = y[train_index].tolist()
        y_test = y[test_index]
        
        # 数据增强
        if use_augmentation and len(train_comments) < 1000:
            logging.info("应用数据增强...")
            train_df = pd.DataFrame({'评论内容': train_comments, '标签': y_train})
            train_df_aug, y_train_aug = augment_dataset(train_df, y_train, target_class=0, augmentation_factor=2)
            train_comments = train_df_aug['评论内容'].tolist()
            y_train = np.array(y_train_aug)
        
        # 创建特征
        logging.info("创建 TF-IDF + 文本统计特征...")
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.9, ngram_range=(1, 2))
        X_train, tfidf_vectorizer = create_enhanced_features(train_comments, tfidf_vectorizer, fit=True)
        X_test, _ = create_enhanced_features(test_comments, tfidf_vectorizer, fit=False)
        
        # SMOTE 过采样
        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train)
        
        # 构建模型
        logging.info("构建集成学习模型...")
        models = build_ensemble_model()
        
        # 训练和评估每个模型
        for model_name, model in models.items():
            logging.info(f"\n训练 {model_name} 模型...")
            
            try:
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 获取预测概率
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = y_pred
                
                # 优化阈值
                if len(np.unique(y_train)) == 2:
                    optimal_thresh = optimize_threshold(y_train, model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_pred)
                    optimal_thresholds[model_name] = optimal_thresh
                    y_pred_optimized = (y_pred_proba >= optimal_thresh).astype(int)
                else:
                    y_pred_optimized = y_pred
                
                # 评估
                result = evaluate_model(y_test, y_pred_optimized, y_pred_proba, model_name)
                fold_results[model_name].append(result)
                
                logging.info(f"{model_name} - F1: {result['f1_score']:.4f}, Accuracy: {result['accuracy']:.4f}")
                
                # 保存最佳模型
                if fold == 0:
                    best_models[model_name] = model
                
            except Exception as e:
                logging.warning(f"{model_name} 训练失败：{e}")
    
    # 计算平均结果
    avg_results = {}
    for model_name, results in fold_results.items():
        if results:
            avg_results[model_name] = {
                'accuracy': np.mean([r['accuracy'] for r in results]),
                'precision': np.mean([r['precision'] for r in results]),
                'recall': np.mean([r['recall'] for r in results]),
                'f1_score': np.mean([r['f1_score'] for r in results])
            }
            if 'auc' in results[0]:
                avg_results[model_name]['auc'] = np.mean([r['auc'] for r in results])
    
    return avg_results, best_models, optimal_thresholds, tfidf_vectorizer

def visualize_enhanced_results(avg_results):
    """可视化增强模型的对比结果"""
    logging.info("\n正在生成对比可视化结果...")
    
    models = list(avg_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, metric in enumerate(metrics):
        values = [avg_results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=colors, alpha=0.8)
        axes[i].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=15)
        
        for bar, value in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
    logging.info("对比图已保存到 enhanced_model_comparison.png")
    plt.show()
    
    # 打印详细结果
    logging.info("\n" + "="*60)
    logging.info("增强模型对比结果")
    logging.info("="*60)
    
    for model in models:
        logging.info(f"\n{model.upper()}:")
        for metric in metrics:
            logging.info(f"  {metric}: {avg_results[model][metric]:.4f}")
        if 'auc' in avg_results[model]:
            logging.info(f"  AUC: {avg_results[model]['auc']:.4f}")
    
    # 找出最佳模型
    best_model = max(avg_results.keys(), key=lambda m: avg_results[m]['f1_score'])
    logging.info(f"\n🏆 最佳模型（基于 F1 分数）: {best_model.upper()}")
    logging.info(f"   F1 Score: {avg_results[best_model]['f1_score']:.4f}")

def main():
    """主函数"""
    try:
        logging.info("正在加载数据...")
        df = pd.read_csv("data/comments.csv")
        fraudulent_df = pd.read_csv("data/fraudulent_comments.csv")
    except FileNotFoundError as e:
        logging.error(f"文件未找到：{e}")
        return
    
    comments = df['评论内容'].tolist()
    fraudulent_comments_set = set(fraudulent_df['评论内容'].tolist())
    y = [1 if comment not in fraudulent_comments_set else 0 for comment in comments]
    
    logging.info(f"数据集大小：{len(df)} 条评论")
    logging.info(f"正常评论：{sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    logging.info(f"刷单评论：{len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # 训练增强模型
    avg_results, best_models, optimal_thresholds, tfidf_vectorizer = train_and_evaluate_enhanced(
        df, y,
        use_augmentation=True,
        use_smote=True,
        use_ensemble=True
    )
    
    # 可视化结果
    visualize_enhanced_results(avg_results)
    
    # 选择最佳模型
    best_model_name = max(avg_results.keys(), key=lambda m: avg_results[m]['f1_score'])
    best_model = best_models[best_model_name]
    
    # 保存最佳模型
    joblib.dump(best_model, f'models_output/best_model_{best_model_name}.pkl')
    joblib.dump(tfidf_vectorizer, f'models_output/tfidf_vectorizer.pkl')
    logging.info(f"\n最佳模型 {best_model_name} 已保存到 models_output/")
    
    # 在完整数据集上预测
    logging.info("\n在完整数据集上进行预测...")
    X_full, _ = create_enhanced_features(comments, tfidf_vectorizer, fit=False)
    predictions = best_model.predict(X_full)
    
    fraudulent_indices = [i for i, pred in enumerate(predictions) if pred == 0]
    detected_fraudulent = [df['评论内容'].iloc[i] for i in fraudulent_indices]
    
    logging.info(f"最佳模型：{best_model_name}")
    logging.info(f"最佳模型平均结果：{avg_results[best_model_name]}")
    logging.info(f"识别出的刷单评论数量：{len(detected_fraudulent)}")
    
    # 混淆矩阵
    y_np = np.array(y)
    cm = confusion_matrix(y_np, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 分类报告
    logging.info("\n分类报告:")
    logging.info(classification_report(y_np, predictions, target_names=['刷单评论', '正常评论']))
    
    logging.info("\n增强模型训练完成！")

if __name__ == "__main__":
    main()
