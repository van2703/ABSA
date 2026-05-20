import pandas as pd

df = pd.read_csv(r"D:\Workspace\b2\Machine learning 1\ABSA\review_aspect.csv")

# Xóa NaN
df = df.dropna()

aspect_cols = [
    "chat_luong",
    "dong_goi",
    "van_chuyen",
    "gia_ca",
    "overall"
]

rows = []

for _, row in df.iterrows():

    comment = str(row["comment"]).strip()

    for aspect in aspect_cols:

        sentiment = str(row[aspect]).strip()

        rows.append({
            "comment": comment,
            "aspect": aspect,
            "sentiment": sentiment
        })

new_df = pd.DataFrame(rows)

print(new_df.head())
print(new_df.shape)

new_df.to_csv("../data/review_aspect_long.csv", index=False, encoding='utf-8-sig')

print("DONE")