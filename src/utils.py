from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """
    Return the project root directory.

    This function assumes that utils.py is inside the src/ folder.
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(directory_path):
    """
    Create a directory if it does not exist.
    """
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def load_csv(file_path):
    """
    Load a CSV file using utf-8-sig encoding.
    """
    return pd.read_csv(file_path, encoding="utf-8-sig")


def save_csv(df, file_path, index=False):
    """
    Save a dataframe to CSV using utf-8-sig encoding.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=index, encoding="utf-8-sig")
    print(f"Saved file to: {file_path}")