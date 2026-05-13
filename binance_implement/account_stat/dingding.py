import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import json
import os

def send_dingtalk_message(message):
    access_token = os.environ["DINGTALK_ACCESS_TOKEN"]
    secret = os.environ["DINGTALK_SECRET"]

    # 获取当前时间戳，单位毫秒
    timestamp = str(round(time.time() * 1000))

    # 拼接待签名字符串：timestamp + "\n" + secret
    string_to_sign = f'{timestamp}\n{secret}'

    # 使用 HMAC-SHA256 算法计算签名
    hmac_code = hmac.new(secret.encode('utf-8'), string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
    # 对结果进行 base64 编码，并进行 URL 编码
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    # 构造请求 URL，将 access_token、timestamp 和 sign 作为参数传入
    url = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}&timestamp={timestamp}&sign={sign}'

    headers = {'Content-Type': 'application/json;charset=utf-8'}
    data = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # print(response.text)

# if name == "main":
#     # 调用函数发送消息
#     send_dingtalk_message("测试消息：钉钉机器人发送成功！")
