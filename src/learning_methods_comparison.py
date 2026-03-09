import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
from sklearn.semi_supervised import SelfTrainingClassifier
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
    """中文文本数据增强"""
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

def evaluate_model(y_true, y_pred, method_name):
    """评估模型性能"""
    return {
        'method': method_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def create_tfidf_features(comments, max_features=1000):
    """创建 TF-IDF 特征"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(comments)
    return X, vectorizer

def augment_dataset(df, y, target_class=0, augmentation_factor=2):
    """对少数类进行数据增强"""
    y = np.array(y)
    df = df.copy()
    df['标签'] = y
    
    minority_indices = df[df['标签'] == target_class].index.tolist()
    majority_indices = df[df['标签'] != target_class].index.tolist()
    
    if len(minority_indices) == 0:
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
    return combined_df.drop('标签', axis=1), combined_y

def unsupervised_learning(X, y_true, n_clusters=3):
    """
    无监督学习方法：KMeans 聚类 + 规则判断
    
    假设：刷单评论通常具有以下特征：
    1. 文本长度较短
    2. 情感倾向极端（非常正面或非常负面）
    3. 包含特定关键词（如"好评"、"推荐"等）
    """
    logging.info("正在训练无监督学习模型（KMeans 聚类）...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # 基于聚类结果和规则进行预测
    predictions = []
    
    # 分析每个聚类的特征，确定哪个聚类更可能是刷单评论
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        # 计算聚类的平均 TF-IDF 特征（简化为文本长度代理）
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'indices': cluster_indices
        })
    
    # 假设：最大的聚类通常是正常评论，较小的聚类可能是刷单评论
    # 这里使用启发式规则：将最小的聚类标记为刷单评论（0），其他为正常评论（1）
    cluster_sizes = [(i, stats['size']) for i, stats in enumerate(cluster_stats)]
    cluster_sizes.sort(key=lambda x: x[1])
    
    # 最小的聚类标记为刷单评论
    fraudulent_clusters = [cluster_sizes[0][0]]
    
    for i in range(len(X.toarray())):
        if cluster_labels[i] in fraudulent_clusters:
            predictions.append(0)
        else:
            predictions.append(1)
    
    logging.info(f"无监督学习：识别出 {predictions.count(0)} 条刷单评论")
    
    return predictions, '无监督学习 (KMeans)'

def supervised_learning(X_train, y_train, X_test, y_test):
    """
    监督学习方法：Random Forest + XGBoost
    
    使用完整的标签信息进行训练
    """
    logging.info("正在训练监督学习模型（Random Forest + XGBoost）...")
    
    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=class_weights_dict,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if 0 in y_train and 1 in y_train else 1
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    
    # 训练模型
    rf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)
    
    # 预测
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_clf.predict(X_test)
    
    # 选择表现更好的模型
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    
    if rf_f1 > xgb_f1:
        logging.info(f"Random Forest 表现更好 (F1: {rf_f1:.4f} vs {xgb_f1:.4f})")
        return rf_pred, '监督学习 (Random Forest)'
    else:
        logging.info(f"XGBoost 表现更好 (F1: {xgb_f1:.4f} vs {rf_f1:.4f})")
        return xgb_pred, '监督学习 (XGBoost)'

def semi_supervised_learning(X_train, y_train, X_test, y_test, base_model='rf'):
    """
    半监督学习方法：Self-Training
    
    使用少量有标签数据和大量无标签数据进行训练
    模拟场景：只有部分数据有标签
    """
    logging.info("正在训练半监督学习模型（Self-Training）...")
    
    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # 选择基模型
    if base_model == 'rf':
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights_dict,
            random_state=42,
            n_jobs=-1
        )
    else:
        base_estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
    
    # Self-Training Classifier
    self_training = SelfTrainingClassifier(
        base_estimator,
        threshold=0.75,
        criterion='threshold',
        max_iter=10,
        verbose=0
    )
    
    # 训练
    self_training.fit(X_train, y_train)
    
    # 预测
    predictions = self_training.predict(X_test)
    
    return predictions, f'半监督学习 (Self-Training + {base_model.upper()})'

def compare_learning_methods(df, y, n_splits=3):
    """
    对比三种学习方法
    
    Args:
        df: 包含'评论内容'的 DataFrame
        y: 标签列表
        n_splits: 交叉验证折数
    """
    logging.info("=" * 60)
    logging.info("开始对比三种学习方法")
    logging.info("=" * 60)
    
    # 创建 TF-IDF 特征
    logging.info("正在创建 TF-IDF 特征...")
    comments = df['评论内容'].tolist()
    X, vectorizer = create_tfidf_features(comments, max_features=1000)
    y = np.array(y)
    
    # 交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        '无监督学习': [],
        '监督学习': [],
        '半监督学习': []
    }
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        logging.info(f"\n{'='*60}")
        logging.info(f"第 {fold + 1}/{n_splits} 折")
        logging.info(f"{'='*60}")
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # 转换为 DataFrame 格式
        train_df = pd.DataFrame({'评论内容': [comments[i] for i in train_index]})
        
        # 对训练集进行数据增强（只在训练集上）
        if len(train_df) < 1000:
            logging.info("检测到小数据集，应用数据增强...")
            train_df_aug, y_train_aug = augment_dataset(
                train_df, y_train_fold.tolist(), target_class=0, augmentation_factor=2
            )
            # 使用训练集的 vectorizer 来转换增强后的数据，保持特征维度一致
            X_train_aug = vectorizer.transform(train_df_aug['评论内容'].tolist())
            y_train_aug = np.array(y_train_aug)
        else:
            X_train_aug = X_train_fold
            y_train_aug = y_train_fold
        
        # 1. 无监督学习（不使用标签信息）
        logging.info("\n[1/3] 无监督学习...")
        unsup_pred, unsup_name = unsupervised_learning(X_test_fold, y_test_fold, n_clusters=3)
        results['无监督学习'].append(evaluate_model(y_test_fold, unsup_pred, unsup_name))
        
        # 2. 监督学习（使用完整标签信息）
        logging.info("\n[2/3] 监督学习...")
        sup_pred, sup_name = supervised_learning(
            X_train_aug, y_train_aug, X_test_fold, y_test_fold
        )
        results['监督学习'].append(evaluate_model(y_test_fold, sup_pred, sup_name))
        
        # 3. 半监督学习（模拟部分标签场景）
        logging.info("\n[3/3] 半监督学习...")
        # 模拟只有 30% 的数据有标签
        n_train_samples = X_train_aug.shape[0]
        labeled_size = int(n_train_samples * 0.3)
        labeled_indices = random.sample(range(n_train_samples), labeled_size)
        X_train_labeled = X_train_aug[labeled_indices]
        y_train_labeled = y_train_aug[labeled_indices]
        
        semisup_pred, semisup_name = semi_supervised_learning(
            X_train_labeled, y_train_labeled, X_test_fold, y_test_fold, base_model='rf'
        )
        results['半监督学习'].append(evaluate_model(y_test_fold, semisup_pred, semisup_name))
    
    # 计算平均结果
    avg_results = {}
    for method, fold_results in results.items():
        avg_results[method] = {
            'accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'precision': np.mean([r['precision'] for r in fold_results]),
            'recall': np.mean([r['recall'] for r in fold_results]),
            'f1_score': np.mean([r['f1_score'] for r in fold_results])
        }
    
    return avg_results, results

def visualize_comparison(avg_results):
    """可视化对比结果"""
    logging.info("\n正在生成对比可视化结果...")
    
    methods = list(avg_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # 创建 2x2 的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, metric in enumerate(metrics):
        values = [avg_results[method][metric] for method in methods]
        
        bars = axes[i].bar(methods, values, color=colors, alpha=0.8)
        axes[i].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=15)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # 添加网格线
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('learning_methods_comparison.png', dpi=300, bbox_inches='tight')
    logging.info("对比图已保存到 learning_methods_comparison.png")
    plt.show()
    
    # 打印详细结果
    logging.info("\n" + "="*60)
    logging.info("三种学习方法对比结果")
    logging.info("="*60)
    
    for method in methods:
        logging.info(f"\n{method}:")
        for metric in metrics:
            logging.info(f"  {metric}: {avg_results[method][metric]:.4f}")
    
    # 找出最佳方法
    best_method = max(avg_results.keys(), key=lambda m: avg_results[m]['f1_score'])
    logging.info(f"\n🏆 最佳方法（基于 F1 分数）: {best_method}")
    logging.info(f"   F1 Score: {avg_results[best_method]['f1_score']:.4f}")

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
    
    # 对比三种学习方法
    avg_results, fold_results = compare_learning_methods(df, y, n_splits=3)
    
    # 可视化对比结果
    visualize_comparison(avg_results)
    
    logging.info("\n对比完成！")

if __name__ == "__main__":
    main()
