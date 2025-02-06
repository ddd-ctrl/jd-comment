import requests
import pandas as pd

# 商品ID，可根据实际情况修改
product_id = "10101991812161"
# 要爬取的总页数
total_pages = 20

# 存储评论数据的列表
comments_list = []

# 循环遍历每一页
for page in range(1, total_pages + 1):
    # 构造请求URL
    url = f"https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1738815087230&body=%7B%22productId%22%3A{product_id}%2C%22score%22%3A0%2C%22sortType%22%3A5%2C%22page%22%3A{page}%2C%22pageSize%22%3A10%2C%22isShadowSku%22%3A0%2C%22rid%22%3A0%2C%22fold%22%3A1%2C%22bbtf%22%3A%22%22%2C%22shield%22%3A%22%22%7D"

    # 设置请求头，模拟浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    try:
        # 发送请求
        response = requests.get(url, headers=headers)
        # 检查响应状态码
        response.raise_for_status()

        # 解析JSON数据
        data = response.json()

        # 提取评论信息
        comments = data.get("comments", [])
        for comment in comments:
            # 提取需要的信息，如评论内容、评论时间、用户昵称等
            content = comment.get("content", "")
            creation_time = comment.get("creationTime", "")
            nickname = comment.get("nickname", "")

            # 将信息添加到列表中
            comments_list.append({
                "nickname": nickname,
                "creation_time": creation_time,
                "content": content
            })

    except requests.RequestException as e:
        print(f"请求出错: {e}")
    except ValueError as e:
        print(f"JSON解析出错: {e}")

# 将评论数据转换为DataFrame
df = pd.DataFrame(comments_list)

# 保存为CSV文件
df.to_csv("jd_comments.csv", index=False, encoding="utf-8-sig")
print("评论数据已保存到 jd_comments.csv")