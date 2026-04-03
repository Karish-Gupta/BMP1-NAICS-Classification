#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10

# GPU is preferred
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Label encoder for unseen givens
class SafeLabelEncoder:
    def __init__(self):
        self.classes_ = []
        self.class_to_idx = {}
        self.unk_idx = -1

    def fit(self, y):
        self.classes_ = list(set(y))
        self.classes_.append('<UNK>')
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.unk_idx = self.class_to_idx['<UNK>']
        return self

    def transform(self, y):
        return np.array([self.class_to_idx.get(str(item), self.unk_idx) for item in y])

    def inverse_transform(self, y):
        return np.array([self.classes_[idx] for idx in y])

    def get_num_classes(self):
        return len(self.classes_)

# Dataset preperation
class SequentialNAICSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, given_codes=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.given_codes = given_codes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
             sample['labels'] = torch.tensor(self.labels[item], dtype=torch.long)
        if self.given_codes is not None:
            sample['given_code'] = torch.tensor(self.given_codes[item], dtype=torch.long)
        return sample

# Sequemtial NAICS model
class SequentialNAICSModel(nn.Module):
    def __init__(self, n_classes, n_given_levels):
        super(SequentialNAICSModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

        self.use_context = n_given_levels > 0
        if self.use_context:
            self.context_embedding = nn.Embedding(n_given_levels, 64)
            input_dim = 768 + 64
        else:
            input_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask, given_code=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        if self.use_context and given_code is not None:
            ctx = self.context_embedding(given_code)
            combined = torch.cat((pooled, ctx), dim=1)
        else:
            combined = pooled

        return self.classifier(combined)

# Takes CSV structure
def process_level_data(level_num):
    file_path = f"d{level_num}_train.csv"
    df = pd.read_csv(file_path)

    df['text_input'] = df['inputs'].astype(str)

    le_target = SafeLabelEncoder()
    df['y'] = le_target.fit(df['predictor'].astype(str)).transform(df['predictor'].astype(str))

    le_given = None
    num_parent_codes = 0
    if 'given' in df.columns:
        le_given = SafeLabelEncoder()
        df['parent_idx'] = le_given.fit(df['given'].astype(str)).transform(df['given'].astype(str))
        num_parent_codes = le_given.get_num_classes()
    else:
        df['parent_idx'] = 0

    return df, le_target, le_given, num_parent_codes

def get_level_metrics(y_true, y_pred, level_name):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"--- Stats for Level {level_name} ---")
    print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f}")
    return {"acc": acc, "f1": f1}

# Training loop to output the model
def train_level(level_to_train, tokenizer):
    df, le_target, le_given, n_parents = process_level_data(level_to_train)
    n_classes = le_target.get_num_classes()

    given_idx = df['parent_idx'].values if 'given' in df.columns else None

    train_ds = SequentialNAICSDataset(df['text_input'].values, df['y'].values, tokenizer, MAX_LEN, given_codes=given_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = SequentialNAICSModel(n_classes, n_parents).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"--- Beginning training Level: {level_to_train} ---")
    for epoch in range(EPOCHS): 
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Send data to GPU
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)
            given_code = batch['given_code'].to(DEVICE, non_blocking=True) if 'given_code' in batch else None

            logits = model(input_ids, attention_mask, given_code)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader):.4f}")

    return model, le_target, le_given

# Evaluate model
def evaluate_full_architecture(models, encoders, test_df, tokenizer):
    print("\n--- Evaluating Full Sequential Architecture ---")

    texts = test_df['inputs'].astype(str).values
    # Takes in true codes
    ground_truths = {level: [] for level in range(1, 6)}

    # Finds real code through model
    full_codes = (test_df['given'].astype(str) + test_df['predictor'].astype(str)).values
    for code in full_codes:
        for level in range(1, 6):
            ground_truths[level].append(code[:level+1])

    test_ds = SequentialNAICSDataset(texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Predictions by level
    level_predictions = {level: [] for level in range(1, 6)}

    for m in models.values():
        m.eval()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)

            batch_given_str = [""] * len(input_ids)

            for level in range(1, 6):
                model = models[level]
                le_target = encoders[level]['target']
                le_given = encoders[level]['given']

                if level == 1:
                    logits = model(input_ids, attention_mask, given_code=None)
                else:
                    given_int = le_given.transform(batch_given_str)
                    given_tensor = torch.tensor(given_int, dtype=torch.long).to(DEVICE, non_blocking=True)
                    logits = model(input_ids, attention_mask, given_code=given_tensor)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                pred_strs = le_target.inverse_transform(preds)

                # Update current batch state and store for stats
                for j in range(len(batch_given_str)):
                    batch_given_str[j] += str(pred_strs[j])
                    level_predictions[level].append(batch_given_str[j])

    # Prints stats
    for level in range(1, 6):
        get_level_metrics(ground_truths[level], level_predictions[level], f"Level {level} ({level+1}-digits)")

    return level_predictions[5], ground_truths[5]

# Run model
if __name__ == "__main__":
    # Ensure HuggingFace tokenizers parallelism warning is suppressed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

    pipeline_models = {}
    pipeline_encoders = {}

    # Train models sequentially 
    for level in range(1, 6):
        model, le_target, le_given = train_level(level, tokenizer)
        pipeline_models[level] = model
        pipeline_encoders[level] = {
            'target': le_target,
            'given': le_given
        }

    eval_df = pd.read_csv("test_set.csv")
    print(eval_df)

    preds, truths = evaluate_full_architecture(pipeline_models, pipeline_encoders, eval_df, tokenizer)

    acc = accuracy_score(truths[:len(preds)], preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truths[:len(preds)], preds, average='weighted', zero_division=0
    )

    print("\n--- Pipeline Evaluation Metrics ---")
    print(f"Overall Pipeline Accuracy:  {acc:.4f}")
    print(f"Overall Pipeline Precision: {precision:.4f}")
    print(f"Overall Pipeline Recall:    {recall:.4f}")
    print(f"Overall Pipeline F1-Score:  {f1:.4f}")

