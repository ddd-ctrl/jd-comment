# d:/Code/jd-comment/jd_comment_spider_fixed.py

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import random
import warnings
warnings.filterwarnings("ignore", category=requests.RequestsDependencyWarning)

# 确保编码正常
try:
    import chardet
except ImportError:
    pass

def fetch_jd_comments_safe(product_id, max_comments=1000):
    """安全版本的京东评论爬虫"""
    
    # 增强的请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': f'https://item.jd.com/{product_id}.html',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    comments = []
    session = requests.Session()
    session.headers.update(headers)
    
    # 添加重试机制
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
    
    for page in range(1, (max_comments // 10) + 10):  # 多尝试几页
        try:
            url = f"https://club.jd.com/comment/productPageComments.action"
            params = {
                'productId': product_id,
                'score': 0,
                'sortType': 5,
                'page': page,
                'pageSize': 10,
                'isShadowSku': 0,
                'fold': 1
            }
            
            response = session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    if 'comments' in data and data['comments']:
                        for comment in data['comments']:
                            comments.append({
                                'nickname': comment.get('nickname', '匿名用户'),
                                'score': comment.get('score', 5),
                                'content': comment.get('content', '').strip(),
                                'creationTime': comment.get('creationTime', ''),
                                'location': comment.get('location', '未知')
                            })
                        
                        print(f"第{page}页成功获取{len(data['comments'])}条评论")
                    
                    else:
                        print(f"第{page}页无评论数据")
                        break
                        
                except json.JSONDecodeError:
                    print(f"第{page}页JSON解析失败")
                    continue
                    
            elif response.status_code == 403:
                print("被限制访问，需要等待或使用代理")
                time.sleep(random.randint(5, 10))
                continue
                
            else:
                print(f"第{page}页请求失败: {response.status_code}")
                
        except Exception as e:
            print(f"第{page}页异常: {e}")
            time.sleep(random.randint(3, 7))
            continue
            
        # 随机延时
        time.sleep(random.uniform(1, 3))
        
        if len(comments) >= max_comments:
            break
    
    return comments[:max_comments]

def save_comments_to_csv(comments, file_path):
    """保存评论到CSV文件"""
    try:
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig解决Excel乱码
            writer = csv.writer(f)
            writer.writerow(['用户', '评分', '评论内容', '评论时间', '用户地址'])
            
            for comment in comments:
                writer.writerow([
                    comment['nickname'],
                    comment['score'],
                    comment['content'].replace('\n', ' ').replace('\r', ' '),  # 清理换行
                    comment['creationTime'],
                    comment['location']
                ])
        
        print(f"成功保存{len(comments)}条评论到 {file_path}")
        return True
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False

# 测试使用
if __name__ == "__main__":
    # 测试商品ID（可以替换为你想爬的商品）
    test_product_id = "100012043978"  # iPhone 15示例
    
    print("开始爬取京东评论...")
    comments = fetch_jd_comments_safe(test_product_id, 100)
    
    if comments:
        save_comments_to_csv(comments, "../data/comments.csv")
        print(f"爬取完成！共获取{len(comments)}条评论")
    else:
        print("未获取到任何评论，请检查商品ID或网络连接")