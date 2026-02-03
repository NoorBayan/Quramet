import os
import argparse
import numpy as np
import pandas as pd
import torch
import pyarabic.araby as araby
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    set_seed
)
from arabert.preprocess import ArabertPreprocessor

# --- Configuration ---
SEED = 42
MAX_LEN = 128
TARGET_COLS = ['Tasrihiyya', 'Makniyya', 'Asliyya', 'Tabiyya', 'Murashaha', 'Mujarrada', 'Mutlaqa']

def preprocess_text(text, model_name):
    """
    Apply specific preprocessing based on the model architecture.
    """
    text = str(text)
    if "arabert" in model_name:
        # AraBERT Preprocessor
        prep = ArabertPreprocessor(model_name=model_name)
        return prep.preprocess(text)
    elif "camelbert" in model_name or "marbert" in model_name.lower():
        # Strip Tashkeel for MARBERT and CamelBERT
        return araby.strip_tashkeel(text)
    else:
        return text

def compute_metrics(eval_pred):
    """
    Compute multi-label metrics: Micro F1, Macro F1, Hamming Loss.
    """
    logits, labels = eval_pred
    # Sigmoid activation
    probs = 1 / (1 + np.exp(-logits))
    # Thresholding at 0.5
    predictions = (probs > 0.5).astype(float)
    
    return {
        "micro_f1": f1_score(labels, predictions, average="micro"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "hamming_loss": hamming_loss(labels, predictions)
    }

def train_model(model_name, data_path, output_dir, epochs=10, batch_size=8):
    print(f"--- Starting Training for: {model_name} ---")
    set_seed(SEED)
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
    except:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        
    print(f"Data loaded. Rows: {len(df)}")

    # 2. Preprocess
    print("Preprocessing text...")
    df['processed_text'] = df['Text_Ayah'].apply(lambda x: preprocess_text(x, model_name))
    df['labels'] = df[TARGET_COLS].values.astype(float).tolist()

    # 3. Split
    train_df, test_df = train_test_split(df[['processed_text', 'labels']], test_size=0.15, random_state=SEED)

    # 4. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["processed_text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize_function, batched=True)

    # 5. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_COLS),
        problem_type="multi_label_classification"
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5 if "arabert" not in model_name else 3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        save_total_limit=1,
        report_to="none"
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # 8. Train & Evaluate
    trainer.train()
    results = trainer.evaluate()
    
    print("\n--- Final Evaluation Results ---")
    for key, val in results.items():
        print(f"{key}: {val:.4f}")
    
    return results

if __name__ == "__main__":
    # You can change the model name here directly or use argparse
    # Options: "aubmindlab/bert-base-arabertv2", "CAMeL-Lab/bert-base-arabic-camelbert-ca", "UBC-NLP/MARBERT"
    
    MODEL_NAME = "UBC-NLP/MARBERT" 
    DATA_PATH = "../data/quranic_metaphor_dataset.csv"
    OUTPUT_DIR = "./marbert_results"
    
    train_model(MODEL_NAME, DATA_PATH, OUTPUT_DIR)
