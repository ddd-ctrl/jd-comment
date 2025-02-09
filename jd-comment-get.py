import requests
import pandas as pd
import time
from datetime import datetime

# 配置参数
product_id = "100039702174"  # 示例有效商品ID（小米手机）
total_pages = 20
delay_seconds = 3  # 每页请求间隔

comments_list = []

for page in range(1, total_pages + 1):
    # 动态生成时间戳
    current_timestamp = int(time.mktime(datetime.now().timetuple())) * 1000
    
    # 构造带动态参数的URL
    url = f"https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t={current_timestamp}&body=%7B%22productId%22%3A{product_id}%2C%22score%22%3A0%2C%22sortType%22%3A5%2C%22page%22%3A{page}%2C%22pageSize%22%3A10%2C%22isShadowSku%22%3A0%2C%22rid%22%3A0%2C%22fold%22%3A1%7D"

    # 完整浏览器头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": f"https://item.jd.com/{product_id}.html",
        "Accept": "application/json",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 检查有效响应
        if 'comments' not in response.json():
            print(f"第 {page} 页无评论数据，可能已达末尾")
            break

        data = response.json()
        for comment in data.get("comments", []):
            comments_list.append({
                "nickname": comment.get("nickname", ""),
                "creation_time": comment.get("creationTime", ""),
                "content": comment.get("content", ""),
                "score": comment.get("score", 0),
                "productColor": comment.get("productColor", ""),
                "productSize": comment.get("productSize", "")
            })

        print(f"已爬取第 {page} 页，累计 {len(comments_list)} 条评论")
        time.sleep(delay_seconds)  # 请求间隔

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)[:50]}...")
        break
    except Exception as e:
        print(f"处理异常: {str(e)[:50]}...")
        continue

if comments_list:
    pd.DataFrame(comments_list).to_csv("jd_comments.csv", index=False, encoding="utf-8-sig")
    print(f"成功保存 {len(comments_list)} 条评论到 jd_comments.csv")
else:
    print("未获取到任何评论，请检查：1.商品ID有效性 2.网络连接 3.反爬限制")