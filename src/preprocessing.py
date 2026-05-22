import re
import pandas as pd


TEENCODE_MAPPING = {
    # Negative / informal Vietnamese
    "ko": "không",
    "k": "không",
    "khum": "không",
    "khong": "không",
    "hok": "không",
    "hong": "không",

    # Product / e-commerce terms
    "sp": "sản phẩm",
    "sanpham": "sản phẩm",
    "spham": "sản phẩm",

    # Shipping terms
    "ship": "giao hàng",
    "shipping": "giao hàng",
    "shipper": "người giao hàng",

    # Common abbreviations
    "dc": "được",
    "đc": "được",
    "okie": "ok",
    "oki": "ok",
    "oke": "ok",
    "okela": "ok",
    "mn": "mọi người",
    "mng": "mọi người",
    "mik": "mình",
    "mk": "mình",
    "m": "mình",

    # Price-related
    "r": "rồi",
    "vs": "với",
}


def normalize_repeated_characters(text: str) -> str:
    """
    Normalize repeated characters in informal Vietnamese reviews.

    Examples:
    - đẹpppp -> đẹp
    - nhanhhh -> nhanh
    - okkkkk -> ok
    - shoppp -> shop
    """
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_repeated_punctuation(text: str) -> str:
    """
    Replace repeated punctuation with a single punctuation mark.

    Examples:
    - !!! -> !
    - ??? -> ?
    - ... -> .
    """
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    return text


def normalize_teencode(text: str) -> str:
    """
    Normalize common Vietnamese teencode and e-commerce abbreviations.
    This function works token by token to avoid replacing characters inside words.
    """
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        normalized_token = TEENCODE_MAPPING.get(token, token)
        normalized_tokens.append(normalized_token)

    return " ".join(normalized_tokens)


def clean_text(text: str) -> str:
    """
    Clean Vietnamese Shopee review text for TF-IDF based machine learning models.

    Steps:
    1. Convert to string
    2. Lowercase
    3. Remove URLs
    4. Normalize repeated characters
    5. Normalize repeated punctuation
    6. Remove unnecessary special characters
    7. Normalize teencode
    8. Remove extra spaces

    Important:
    This function does NOT remove negation words such as:
    không, chưa, chẳng, kém, tệ, lỗi, chậm, đắt.
    """

    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove email-like strings if any
    text = re.sub(r"\S+@\S+", " ", text)

    # Normalize repeated characters before teencode mapping
    text = normalize_repeated_characters(text)

    # Normalize repeated punctuation
    text = normalize_repeated_punctuation(text)

    # Replace some punctuation with spaces
    text = re.sub(r"[,\.;:\(\)\[\]\{\}\"“”‘’]", " ", text)

    # Keep Vietnamese letters, English letters, numbers, and spaces
    # Remove emojis and unusual symbols
    text = re.sub(
        r"[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ\s]",
        " ",
        text,
    )

    # Remove extra spaces before teencode
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize teencode token by token
    text = normalize_teencode(text)

    # Remove extra spaces again because some mapping has multiple words
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_text_preprocessing(df: pd.DataFrame, text_column: str = "comment") -> pd.DataFrame:
    """
    Apply clean_text to a dataframe and create a new column clean_comment.
    """
    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe.")

    df["clean_comment"] = df[text_column].apply(clean_text)

    return df