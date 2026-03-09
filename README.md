# 京东评论刷单检测系统 🛒

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于机器学习的京东评论刷单行为检测系统，集成多种先进技术和模型。

---

## 📁 项目结构

```
jd-comment/
├── src/                        # 源代码目录
│   ├── main.py                # 主程序：集成学习模型 ⭐
│   ├── learning_methods_comparison.py  # 学习方法对比实验 🎓
│   ├── spider.py              # 京东评论爬虫
│   └── main_backup.py         # 旧版本代码（保留参考）
├── data/                       # 数据目录
│   ├── comments.csv           # 原始评论数据
│   └── fraudulent_comments.csv # 标注的刷单评论
├── models_output/              # 模型输出目录
│   ├── best_model_*.pkl       # 训练好的模型
│   └── tfidf_vectorizer.pkl   # TF-IDF 向量化器
├── docs/                       # 文档目录
│   ├── learning_methods_research.md    # 学习方法研究报告
│   ├── PROJECT_STRUCTURE.md            # 项目结构说明
│   └── *.png                    # 可视化对比图表
├── requirements.txt            # 依赖包列表
└── README.md                   # 项目说明文档
```

---

## ✨ 核心功能

### 主程序 (`main.py`)

#### 1. **集成学习模型** 🎯
- **VotingClassifier**：5 模型软投票融合
  - Random Forest
  - XGBoost
  - LightGBM
  - Logistic Regression
  - SVM
- **StackingClassifier**：多层堆叠集成
  - 第一层：RF + XGBoost + LightGBM
  - 第二层：Logistic Regression (meta-learner)

#### 2. **特征工程** 📊
- **TF-IDF 特征** (1000 维)
  - 自动提取文本关键词特征
  - n-gram 范围：(1, 2)
  
- **文本统计特征** (13 维)
  - 文本长度、词数、平均词长
  - 标点符号统计（逗号、句号、感叹号、问号）
  - 情感词比例（正面/负面）
  - 刷单关键词检测
  - 词汇多样性指数
  - 描述性词语数量

#### 3. **数据增强** 🔄
- 随机删除（只删除单字词）
- 随机交换（相邻词语交换）
- 语义保持策略
- 仅在训练集应用（避免数据泄露）

#### 4. **不平衡处理** ⚖️
- **SMOTE 过采样**：合成少数类样本
- **类别权重**：平衡分类器权重
- 自动调整参数

#### 5. **评估优化** 📈
- **5 折分层交叉验证**：更可靠的评估
- **阈值优化**：自动寻找最佳分类阈值
- **多指标评估**：
  - Accuracy（准确率）
  - Precision（精确率）
  - Recall（召回率）
  - F1 Score（F1 分数）
  - AUC（ROC 曲线下面积）

#### 6. **可视化** 📊
- 模型性能对比图（2x2）
- 混淆矩阵
- 分类报告

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包**：
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.0.0
xgboost>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
jieba>=0.42.0
imbalanced-learn>=0.10.0
lightgbm>=3.3.0
```

### 2. 爬取评论（可选）

```bash
cd src
python spider.py
```

### 3. 训练模型 ⭐

```bash
cd src
python main.py
```

### 4. 学习方法对比实验（可选）

```bash
cd src
python learning_methods_comparison.py
```

---

## 📊 性能表现

### 预期性能指标

| 模型 | F1 Score | Accuracy | 特点 |
|------|---------|----------|------|
| **Stacking** | ~0.98+ | ~0.98 | 性能最优 ⭐ |
| **Voting** | ~0.97+ | ~0.97 | 稳定可靠 |
| **LightGBM** | ~0.97+ | ~0.97 | 训练快速 |
| **XGBoost** | ~0.96+ | ~0.96 | 经典强效 |
| **Random Forest** | ~0.96+ | ~0.96 | 解释性好 |

### 不同学习方法对比

| 方法 | F1 Score | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|----------|
| **监督学习** | ~0.98 | 性能最优 | 需要充足标签 | 有标注数据 ✅ |
| **半监督学习** | ~0.90 | 仅需 30% 标签 | 可能传播错误 | 标注成本高 |
| **无监督学习** | ~0.70 | 不需要标签 | 性能较低 | 探索性分析 |

---

## 📈 输出示例

### 训练日志
```
2026-03-09 22:50:55 - INFO - 数据集大小：990 条评论
2026-03-09 22:50:55 - INFO - 正常评论：871 (88.0%)
2026-03-09 22:50:55 - INFO - 刷单评论：119 (12.0%)
2026-03-09 22:50:56 - INFO - 应用数据增强...
2026-03-09 22:50:56 - INFO - 生成了 101 条增强样本
2026-03-09 22:50:56 - INFO - 正在应用 SMOTE 过采样...
2026-03-09 22:50:56 - INFO - SMOTE 后样本数：1394 (原始：798)
2026-03-09 22:50:58 - INFO - voting - F1: 0.9758, Accuracy: 0.9747
2026-03-09 22:50:58 - INFO - stacking - F1: 0.9812, Accuracy: 0.9789
...
2026-03-09 22:51:30 - INFO - 🏆 最佳模型（基于 F1 分数）: STACKING
2026-03-09 22:51:30 - INFO -    F1 Score: 0.9823
```

### 保存的文件
- `models_output/best_model_stacking.pkl` - 最佳模型
- `models_output/tfidf_vectorizer.pkl` - 向量化器
- `docs/enhanced_model_comparison.png` - 对比图表
- 混淆矩阵（自动显示）

---

## 🎓 详细文档

### 学习方法对比研究
查看 [`docs/learning_methods_research.md`](docs/learning_methods_research.md)：
- 三种学习方法的实现原理
- 实验设计和流程
- 结果分析和结论
- 使用建议和场景

### 项目结构说明
查看 [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)：
- 文件夹结构详解
- 文件功能说明
- 使用建议

---

## 🔧 自定义配置

### 修改 TF-IDF 参数
```python
tfidf_params = {
    'max_features': 1000,    # 最大特征数
    'min_df': 2,             # 最小文档频率
    'max_df': 0.9,           # 最大文档频率
    'ngram_range': (1, 2)    # n-gram 范围
}
```

### 调整数据增强
```python
# 在 main.py 中
use_augmentation = True      # 是否使用数据增强
augmentation_factor = 2      # 增强倍数
```

### 修改 SMOTE 参数
```python
# 在 main.py 中
use_smote = True             # 是否使用 SMOTE
k_neighbors = 5              # SMOTE 邻居数
```

### 调整集成模型权重
```python
# 在 build_ensemble_model() 中
weights=[2, 2, 2, 1, 1]     # RF, XGB, LGB, LR, SVM 的权重
```

### 修改交叉验证折数
```python
# 在 train_and_evaluate_enhanced() 中
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# n_splits=5 或 10
```

---

## ⚠️ 注意事项

1. **数据泄露问题**
   - ✅ 数据增强只在训练集上进行
   - ✅ 测试集保持原始数据
   - ✅ 交叉验证中每折独立增强

2. **类别不平衡**
   - 刷单评论通常占少数（~12%）
   - ✅ 使用 SMOTE + 类别权重双重处理
   - ✅ 评估时关注 F1 分数而非准确率

3. **特征维度一致性**
   - ✅ 使用相同的 vectorizer 转换测试集
   - ✅ 确保训练集和测试集特征维度一致

4. **随机性**
   - ✅ 设置 `random_state` 确保可重复性
   - ✅ 多次运行取平均值更可靠

5. **训练时间**
   - 集成模型训练时间较长（~30-60 秒）
   - 建议使用 GPU 加速 LightGBM/XGBoost

---

## 📊 性能优化建议

### 小数据集（<1000 条）
- ✅ 使用数据增强
- ✅ 使用 SMOTE
- ✅ 增加交叉验证折数（5-10 折）
- ✅ 简化模型参数

### 大数据集（>10000 条）
- ✅ 减少数据增强
- ✅ 使用 LightGBM（更快）
- ✅ 减少交叉验证折数（3-5 折）
- ✅ 增加模型复杂度

### 超快模式（牺牲少量精度）
```python
# 修改 main.py
use_smote = False            # 跳过 SMOTE
use_ensemble = False         # 只用单一模型
kf = StratifiedKFold(n_splits=3)  # 减少折数
```

---

## 🎯 技术亮点

1. **集成学习**：融合 5 个模型的优势
2. **特征工程**：TF-IDF + 13 维文本统计特征
3. **不平衡处理**：SMOTE + 类别权重
4. **交叉验证**：5 折分层 CV，结果更可靠
5. **阈值优化**：自动寻找最佳分类点
6. **数据增强**：语义保持的中文文本增强
7. **可视化**：全方位的性能展示

---

## 📚 参考文献

1. Breiman, L. (2001). Random Forests.
2. Chen, T., & Guestrin, C. (2016). XGBoost.
3. Chawla, N. V., et al. (2002). SMOTE.
4. Wolpert, D. H. (1992). Stacked Generalization.
5. Zhu, X., & Goldberg, A. B. (2009). Semi-Supervised Learning.

---

## 📄 许可证

MIT License

---

## 👥 作者

如有问题，请提 Issue 或联系作者。

---

**最后更新**: 2026-03-09  
**版本**: 3.0 (统一版)  
**推荐模型**: StackingClassifier (F1 ≈ 0.98+)
