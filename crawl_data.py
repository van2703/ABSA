import requests
import time
import pandas as pd

url = "https://shopee.vn/api/v2/item/get_ratings"

headers = {
    "user-agent": "Mozilla/5.0",
    "referer": "https://shopee.vn/",
    "cookie": "PASTE_COOKIE"
}

params = {
    "filter": 0,
    "flag": 1,
    "limit": 50,
    "offset": 0,
    "type": 0,
    "shopid": 100758171,
    "itemid": 22669818842
}

all_data = []

for i in range(20):
    params["offset"] = i * 50
    
    res = requests.get(url, params=params, headers=headers)
    
    try:
        data = res.json()
    except:
        print("❌ lỗi JSON")
        break

    if "data" not in data:
        print("⚠️ bị block → nghỉ 10s")
        time.sleep(10)
        continue

    ratings = data["data"]["ratings"]

    if not ratings:
        break

    for r in ratings:
        all_data.append(r["comment"])

    print(f"✅ Page {i+1} | total: {len(all_data)}")

    time.sleep(1.5)

df = pd.DataFrame(all_data, columns=["comment"])
df.to_csv("reviews.csv", index=False, encoding="utf-8-sig")