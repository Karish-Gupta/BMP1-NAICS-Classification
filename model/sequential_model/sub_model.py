import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
import joblib
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from model.sequential_model.dataset import NAICSHierarchyDataset
import pandas as pd

class SequentialSubModel(nn.Module):
    def __init__(self, n_classes):
        super(SequentialSubModel, self).__init__()
        # Load ModernBERT-base (768 hidden size)
        self.bert = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        
        # Simple but effective classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (usually the first token)
        # ModernBERT is optimized for long contexts and rotary embeddings
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(pooled_output)

    def train_full_pipeline(model, train_loader, val_loader, epochs, lr, device, model_name):
        """
        All-in-one training function for a NAICS sub-model.
        """
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Scheduler setup
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps), 
            num_training_steps=total_steps
        )

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # INTERNAL TRAIN LOOP
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for batch in train_pbar:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(ids, mask)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, pred = outputs.max(1)
                train_total += targets.size(0)
                train_correct += pred.eq(targets).sum().item()
                
                train_pbar.set_postfix({'loss': loss.item(), 'acc': 100.*train_correct/train_total})

            # INTERNAL VALIDATION LOOP
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    targets = batch['labels'].to(device)

                    outputs = model(ids, mask)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += pred.eq(targets).sum().item()

            val_acc = 100. * val_correct / val_total
            print(f"Epoch {epoch} Result: Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

            # Save Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{model_name}_best.pt")
                print(f"Saved new best model: {val_acc:.2f}%")

        return model

    def preprocess(train_df, val_df, tokenizer, step_name, batch_size=32):
        # Initialize and Fit Label Encoder
        le = LabelEncoder()
        # Combine both to ensure all possible codes are mapped to an ID
        all_next_codes = pd.concat([train_df['next'], val_df['next']]).astype(str)
        le.fit(all_next_codes)
        
        # Transform the labels
        train_df['label_idx'] = le.transform(train_df['next'].astype(str))
        val_df['label_idx'] = le.transform(val_df['next'].astype(str))
        
        # Save the encoder for inference later
        joblib.dump(le, f"{step_name}_encoder.pkl")
        
        # Create Datasets
        train_ds = NAICSHierarchyDataset(train_df, tokenizer)
        val_ds = NAICSHierarchyDataset(val_df, tokenizer)
        
        # Create DataLoaders
        # Use num_workers > 0 to let the CPU tokenize while the GPU trains
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, len(le.classes_)