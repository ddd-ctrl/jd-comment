import requests
from bs4 import BeautifulSoup
import json
import csv

def fetch_jd_comments(product_id, page=1, page_count=1000):
    """
    获取京东商品评论数据
    :param product_id: 商品ID
    :param page: 起始评论页码
    :param page_count: 要爬取的页数
    :return: 评论数据列表
    """
    comments = []
    for i in range(page, page + page_count):
        url = f"https://club.jd.com/comment/productPageComments.action?productId={product_id}&score=0&sortType=5&page={i}&pageSize=10"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            comments.extend(data['comments'])
        else:
            print(f"Failed to fetch comments: {response.status_code}")
    return comments

def save_comments_to_file(comments, file_path):
    """
    将评论数据保存到CSV文件
    :param comments: 评论数据列表
    :param file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入CSV文件的表头
        writer.writerow(['用户', '评分', '评论内容', '评论时间', '用户地址'])
        for comment in comments:
            # 写入每行数据
            writer.writerow([
                comment['nickname'],
                comment['score'],
                comment['content'],
                comment['creationTime'],
                comment.get('location', '未知')
            ])

if __name__ == "__main__":
    product_id = "100080808860"  # 示例商品ID
    comments = fetch_jd_comments(product_id, page_count=1000)
    save_comments_to_file(comments, "d:\\jd-comment\\comments.csv")  # 修改文件扩展名为.csv
    print("评论数据已保存到 comments.csv")