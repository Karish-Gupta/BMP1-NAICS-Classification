import torch
from torch.utils.data import Dataset

class NAICSHierarchyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        # PERFORMANCE: Convert columns to lists once. 
        # Accessing a list by index is O(1), whereas .iloc is significantly slower.
        self.inputs = dataframe['inputs'].astype(str).tolist()
        self.prev = dataframe['prev'].fillna("").astype(str).tolist()
        self.labels = dataframe['label_idx'].tolist()
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Format the instruction-style string
        # Result: "Context NAICS: 31 | BUSINESS: Zoa Energy..."
        combined_text = f"Context NAICS: {self.prev[idx]} | {self.inputs[idx]}"

        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0), # Remove batch dim
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }