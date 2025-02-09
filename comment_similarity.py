import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


# 1. 数据读取
def read_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        comments = df['content'].tolist()
        creation_times = df['creation_time'].tolist()
        nicknames = df['nickname'].tolist()
        ip_addresses = df['ip_address'].tolist()
        return comments, creation_times, nicknames, ip_addresses
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
        return [], [], [], []
    except KeyError as e:
        print(f"CSV 文件中缺少必要的列: {e}")
        return [], [], [], []


# 2. 文本预处理（分词）
def preprocess_text(comments):
    processed_comments = []
    for comment in comments:
        # 使用 jieba 进行分词
        words = jieba.lcut(comment)
        processed_comments.append(" ".join(words))
    return processed_comments


# 3. 特征提取
def extract_features(processed_comments):
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(processed_comments)
    return feature_matrix


# 4. 相似度计算
def calculate_similarity(feature_matrix):
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix


# 5. 结果分析
def analyze_similarity(similarity_matrix, threshold=0.8):
    similar_pairs = []
    num_comments = len(similarity_matrix)
    for i in range(num_comments):
        for j in range(i + 1, num_comments):
            if similarity_matrix[i][j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i][j]))
    return similar_pairs


# 计算高相似度评论数量占比
def calculate_similar_ratio(similar_pairs, total_comments):
    if total_comments == 0:
        return 0
    similar_comment_ids = set()
    for pair in similar_pairs:
        i, j, _ = pair
        similar_comment_ids.add(i)
        similar_comment_ids.add(j)
    ratio = len(similar_comment_ids) / total_comments
    return ratio


# 分析 IP 地址
def analyze_ip_addresses(similar_pairs, ip_addresses):
    if not similar_pairs or not ip_addresses:
        return 0
    similar_ip_addresses = []
    for pair in similar_pairs:
        i, j, _ = pair
        similar_ip_addresses.extend([ip_addresses[i], ip_addresses[j]])
    unique_ips = set(similar_ip_addresses)
    ip_ratio = len(unique_ips) / len(similar_ip_addresses)
    if ip_ratio < 0.5:  # 可根据实际情况调整阈值
        print("高相似度评论来自少数 IP 地址，可能存在刷单行为。")
    return ip_ratio


# 分析时间分布
def analyze_time_distribution(similar_pairs, creation_times):
    if not similar_pairs or not creation_times:
        return None
    similar_comment_times = []
    for pair in similar_pairs:
        i, j, _ = pair
        similar_comment_times.extend([creation_times[i], creation_times[j]])
    similar_comment_times.sort()
    if similar_comment_times:
        first_time = datetime.strptime(similar_comment_times[0], '%Y-%m-%d')
        last_time = datetime.strptime(similar_comment_times[-1], '%Y-%m-%d')
        time_diff = (last_time - first_time).total_seconds()
        if time_diff < 3600:  # 可根据实际情况调整时间阈值（这里设为 1 小时）
            print("高相似度评论集中在短时间内发布，可能存在刷单行为。")
        return time_diff
    return None


# 分析评论者账号
def analyze_nicknames(similar_pairs, nicknames):
    if not similar_pairs or not nicknames:
        return 0
    similar_comment_nicknames = []
    for pair in similar_pairs:
        i, j, _ = pair
        similar_comment_nicknames.extend([nicknames[i], nicknames[j]])
    unique_nicknames = set(similar_comment_nicknames)
    nickname_ratio = len(unique_nicknames) / len(similar_comment_nicknames)
    if nickname_ratio < 0.5:  # 可根据实际情况调整阈值
        print("高相似度评论来自少数账号，可能存在刷单行为。")
    return nickname_ratio


# 主函数
def main(file_path, output_file):
    # 读取数据
    comments, creation_times, nicknames, ip_addresses = read_csv_data(file_path)
    total_comments = len(comments)

    if total_comments == 0:
        print("未读取到有效评论数据，请检查文件内容。")
        return

    # 文本预处理
    processed_comments = preprocess_text(comments)
    # 特征提取
    feature_matrix = extract_features(processed_comments)
    # 相似度计算
    similarity_matrix = calculate_similarity(feature_matrix)
    # 结果分析
    similar_pairs = analyze_similarity(similarity_matrix)

    # 计算高相似度评论数量占比
    similar_ratio = calculate_similar_ratio(similar_pairs, total_comments)

    # 分析高相似度评论的时间分布
    time_diff = analyze_time_distribution(similar_pairs, creation_times)

    # 分析高相似度评论的评论者账号
    nickname_ratio = analyze_nicknames(similar_pairs, nicknames)

    # 分析 IP 地址
    ip_ratio = analyze_ip_addresses(similar_pairs, ip_addresses)

    # 标记重复评论
    duplicate_comments = [False] * total_comments
    for pair in similar_pairs:
        i, j, _ = pair
        duplicate_comments[i] = True
        duplicate_comments[j] = True

    # 将结果保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"高相似度评论数量占比: {similar_ratio * 100:.2f}%\n")
        if time_diff is not None:
            f.write(f"高相似度评论发布时间间隔: {time_diff} 秒\n")
        f.write(f"高相似度评论评论者账号独特率: {nickname_ratio * 100:.2f}%\n")
        f.write(f"高相似度评论 IP 地址独特率: {ip_ratio * 100:.2f}%\n")
        f.write("-" * 50 + "\n")
        for pair in similar_pairs:
            i, j, similarity = pair
            f.write(f"评论 {i} 和评论 {j} 的相似度为: {similarity}\n")
            f.write(f"评论 {i}: {comments[i]}\n")
            f.write(f"评论 {j}: {comments[j]}\n")
            f.write("-" * 50 + "\n")

        # 写入重复评论标记
        f.write("重复评论标记:\n")
        for idx, is_duplicate in enumerate(duplicate_comments):
            f.write(f"评论 {idx}: {'是' if is_duplicate else '否'}\n")

    print(f"结果已保存到 {output_file}")
    if similar_ratio > 0.2:  # 可根据实际情况调整阈值
        print("高相似度评论占比过高，可能存在刷单行为。")


if __name__ == "__main__":
    file_path = "jd-comments.csv"  # 替换为你的 CSV 文件路径
    output_file = "similarity_results.txt"  # 输出文件路径
    main(file_path, output_file)