import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

class NAICSDataset(Dataset):
    def __init__(self, df, model_name="answerdotai/ModernBERT-large", max_len=128, encoders=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.df = df.reset_index(drop=True)
        
        # If encoders are provided (for validation/test), use them. 
        # Otherwise (for training), create and fit new ones.
        if encoders:
            self.enc_s = encoders['s']
            self.enc_ss = encoders['ss']
            self.enc_i = encoders['i']
        else:
            self.enc_s = LabelEncoder().fit(self.df["sector_label"].astype(str))
            self.enc_ss = LabelEncoder().fit(self.df["subsector_label"].astype(str))
            self.enc_i = LabelEncoder().fit(self.df["industry_label"].astype(str))

        # Transform strings to the shared integer indices
        self.s_indices = self.enc_s.transform(self.df["sector_label"].astype(str))
        self.ss_indices = self.enc_ss.transform(self.df["subsector_label"].astype(str))
        self.i_indices = self.enc_i.transform(self.df["industry_label"].astype(str))

    def get_encoders(self):
        """Helper to pass encoders from train dataset to test dataset"""
        return {'s': self.enc_s, 'ss': self.enc_ss, 'i': self.enc_i}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df["inputs"].iloc[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "context_input": {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            },
            "labels": {
                "sector": torch.tensor(self.s_indices[idx], dtype=torch.long),
                "subsector": torch.tensor(self.ss_indices[idx], dtype=torch.long),
                "industry": torch.tensor(self.i_indices[idx], dtype=torch.long)
            }
        }