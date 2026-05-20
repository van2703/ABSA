import pandas as pd


df = pd.read_csv("../data/absa_preprocessed.csv")

# Tạo input cho model
df["text_input"] = (
    df["clean_comment"]
    + " aspect_" 
    + df["aspect"]
)

print(df[["clean_comment", "aspect", "text_input"]].head())

# Save
df.to_csv("../data/absa_final.csv", index=False, encoding='utf-8-sig')

print("DONE")