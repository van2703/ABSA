import re
import regex
import emoji

from underthesea import word_tokenize


def normalize_repeated_chars(text):

    return regex.sub(r'(.)\1{2,}', r'\1', text)


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = emoji.replace_emoji(text, replace='')

    text = normalize_repeated_chars(text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    text = word_tokenize(text)

    return text
sample = "SP ĐẸP VÃIIII 😭😭 giao nhanhhhhh!!!!"

print(preprocess_text(sample))