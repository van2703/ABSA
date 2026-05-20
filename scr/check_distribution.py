import pandas as pd

df = pd.read_csv(r"D:\Workspace\b2\Machine learning 1\ABSA\data\absa_final.csv")

print(df["sentiment"].value_counts())
df["text_input"] = (
    df["clean_comment"]
    + " aspect_" 
    + df["aspect"]
)