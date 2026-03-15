from transformers import AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

class NAICSMultiHeadClassifier(nn.Module):
    def __init__(self, 
                 model_name="answerdotai/ModernBERT-large", 
                 num_sectors=24, 
                 num_subsectors=98, 
                 num_industries=1066):
        super(NAICSMultiHeadClassifier, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.refinement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.sector_head = nn.Linear(hidden_size, num_sectors)
        self.subsector_head = nn.Linear(hidden_size, num_subsectors)
        self.industry_head = nn.Linear(hidden_size, num_industries)

    def forward(self, context_input):
        outputs = self.encoder(**context_input)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        refined_features = self.refinement(pooled_output)
        
        return {
            "sector": self.sector_head(refined_features),
            "subsector": self.subsector_head(refined_features),
            "industry": self.industry_head(refined_features)
        }

    def fit(self, train_loader, val_loader, loss_fn, epochs=3, lr=2e-5, device="cuda"):
        """Main training loop method"""
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in pbar:
                inputs = {k: v.to(device) for k, v in batch['context_input'].items()}
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                
                optimizer.zero_grad()
                preds = self(inputs)
                
                # Use your HierarchicalLoss class
                loss = loss_fn(preds, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Run validation after each epoch
            self.evaluate(val_loader, device)

    @torch.no_grad()
    def evaluate(self, val_loader, device="cuda"):
        """Validation method to calculate hierarchical Accuracy, Precision, Recall, and F1"""
        self.eval()
        
        # Storage for all predictions and true labels to calculate global metrics
        results = {
            "sector": {"preds": [], "labels": []},
            "subsector": {"preds": [], "labels": []},
            "industry": {"preds": [], "labels": []}
        }
        
        pbar = tqdm(val_loader, desc="Validating")
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch['context_input'].items()}
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            preds = self(inputs)
            
            for level in results.keys():
                # Convert logits to class indices and move to CPU for sklearn
                p = preds[level].argmax(1).cpu().numpy()
                l = labels[level].cpu().numpy()
                
                results[level]["preds"].extend(p)
                results[level]["labels"].extend(l)
                
        print(f"\n" + "="*50)
        print(f"{'Level':<12} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
        print("-" * 50)

        for level in ["sector", "subsector", "industry"]:
            y_true = np.array(results[level]["labels"])
            y_pred = np.array(results[level]["preds"])
            
            # Calculate Accuracy
            acc = (y_true == y_pred).mean() * 100
            
            # Calculate Precision, Recall, and F1 (weighted handles class imbalance)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            print(f"{level.capitalize():<12} | {acc:>6.2f}% | {precision:>6.2f} | {recall:>6.2f} | {f1:>6.2f}")
        
        print("="*50 + "\n")
        
        # Return the industry F1 score specifically, often used for model checkpointing
        return f1