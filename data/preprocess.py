import re
import pandas as pd
from sklearn.model_selection import train_test_split

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9.?! ]+", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# def load_and_preprocess_data(data_path):
#     data = pd.read_csv(data_path)
#     data['text'] = data['text'].apply(clean_text)
#     data['summary'] = data['summary'].apply(clean_text)
#     train_data, val_data = train_test_split(data, test_size=0.2)
#     return train_data, val_data

### IMPORT DATASET ###

from datasets import load_dataset

# cnn_dailymail dataset https://huggingface.co/datasets/abisee/cnn_dailymail
def load_cnn_dailymail():   
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    return dataset
    
dataset = load_cnn_dailymail()

# print("Article:")
# print(dataset["train"])
# print("Article:")
# print(dataset["train"]["article"][0])
# print("Summary:")
# print(dataset["train"]["highlights"][0])

### TOKENIZATION ###

from datasets import load_dataset
from transformers import AutoTokenizer


def preprocess_cnn_dailymail(
    tokenizer_name="t5-small", max_input_length=512, max_output_length=150, fraction=0.25
):
    # Carica il dataset
    dataset = load_cnn_dailymail()

    # Uso solo una frazione del dataset
    def take_fraction(dataset_split, fraction):
        total_size = len(dataset_split)
        subset_size = int(total_size * fraction)
        return dataset_split.select(range(subset_size))

    dataset["train"] = take_fraction(dataset["train"], fraction)
    dataset["validation"] = take_fraction(dataset["validation"], fraction)
    dataset["test"] = take_fraction(dataset["test"], fraction)

    # Inizializza il tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(example):
        inputs = tokenizer(
            example["article"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            example["highlights"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    # Tokenizza il dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets, tokenizer


dataset = preprocess_cnn_dailymail(fraction=0.25)
# print("Esempio tokenizzato:")
# print(dataset["train"][0])