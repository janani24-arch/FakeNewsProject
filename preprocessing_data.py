import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

# -----------------------------------------------
# BASIC CLEANING FUNCTIONS
# -----------------------------------------------
def clean_text(text):

    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove mentions (@user)
    text = re.sub(r"@\w+", "", text)

    # remove hashtags symbol only (#text → text)
    text = re.sub(r"#", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------------------
# LOAD + APPLY CLEANING
# -----------------------------------------------
df = pd.read_csv("datasets_folder/weibo_dataset_en.csv")   # your raw file

df["clean_text"] = df["text_en"].apply(clean_text)

# -----------------------------------------------
# SAVE CLEANED DATA
# -----------------------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("Cleaning complete. Rows:", len(df))
print(df.head())

# load your dataset
df = pd.read_csv("cleaned_data.csv")

# split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,        # 20% test
    random_state=42,      # fixed for reproducibility
    stratify=df["label"]  # keeps label distribution balanced
)

# save them
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train size :", len(train_df))
print("Test size  :", len(test_df))
