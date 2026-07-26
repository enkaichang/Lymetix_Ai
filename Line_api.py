import requests
import json

def send_line_message(msg):
    # 請將 YOUR_CHANNEL_ACCESS_TOKEN_HERE 替換為您的 Channel Access Token
    # channel_access_token = 'srtjYGjSGOzlyQvhbb3pSH/nm/NVdSYOXNYJEyr0YZgoZO/rYqQI/zdBc9SC+e8rF0Dl2CJUXyBlwfy8+FjkQFvls4PvQcaGBj5s/ayBjN9FGK3Qm1o+z331G9QuCneCeO98uS0lddaHE/kFr0Uw+AdB04t89/1O/w1cDnyilFU='
    channel_access_token = 'M7h4Xv0XMGo1iFtg5Q/v8wWSptZrIi6re0pOmHFwbQ0EgozWrFNg+2OKiOTlbgBVBx7WEbL3kTbVlbeVW4X1TiAkqKIdu6xUN5ANJkPx9nCQA8pWpZ4Hxjrp+yl1qb8TSMn5kP/7mZsWvaI3w/4EAwdB04t89/1O/w1cDnyilFU='

    # API 端點
    url = 'https://api.line.me/v2/bot/message/broadcast'

    # 訊息主體
    # 您可以根據文件中的格式，替換或增加不同的訊息物件
    # 這裡是一個傳送文字訊息和貼圖的範例
    message_body = {
    "messages": [
        {
        "type": "text",
        "text": msg
        },
    ],
    "notificationDisabled": True
    }

    # 要求標頭
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }

    try:
        # 執行 POST 請求
        response = requests.post(url, headers=headers, data=json.dumps(message_body))
        response.raise_for_status()  # 如果請求不成功，將會引發 HTTPError

        # 顯示回應內容
        print("訊息傳送成功！")
        print(f"狀態碼: {response.status_code}")
        print(f"回應內容: {response.text}")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP 錯誤: {err}")
        print(f"回應狀態碼: {response.status_code}")
        print(f"回應內容: {response.text}")
    except Exception as err:
        print(f"發生其他錯誤: {err}")

def get_stock_result():
    import use_model
    result = use_model.main()
    predicter = use_model.stockStockPredictor()
    msg = predicter.join_txt2msg(result)
    send_line_message(msg)

def in_time_send():
    from datetime import datetime
    now = datetime.now()
    while True:
        if now.hour == 19 and now.minute == 36:
            get_stock_result()
            print("sent")
            break

get_stock_result()