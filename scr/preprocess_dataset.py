import pandas as pd

from preprocess import preprocess_text


df = pd.read_csv(r"D:\Workspace\b2\Machine learning 1\ABSA\data\review_aspect_long.csv")

df["clean_comment"] = df["comment"].apply(preprocess_text)

print(df.head())

df.to_csv("../data/absa_preprocessed.csv", index=False, encoding='utf-8-sig')

print("DONE")