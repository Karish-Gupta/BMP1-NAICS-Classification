import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from model.multi_head_model.model import NAICSMultiHeadClassifier
from model.multi_head_model.dataset import NAICSDataset
from model.multi_head_model.loss import HierarchicalLoss

# Load data
train_df = pd.read_csv("data/split_data_preprocessed/train_data_with_summaries.csv", encoding="latin-1") 
test_df = pd.read_csv("data/split_data_preprocessed/test_data_with_summaries.csv", encoding="latin-1")
val_df = pd.read_csv("data/split_data_preprocessed/val_data_with_summaries.csv", encoding="latin-1")

# Initialize Datasets
train_dataset = NAICSDataset(train_df, model_name="answerdotai/ModernBERT-large")

# Extract the fitted encoders once to share with both val and test
shared_encoders = train_dataset.get_encoders()

val_dataset = NAICSDataset(
    val_df, 
    model_name="answerdotai/ModernBERT-large",
    encoders=shared_encoders
)

test_dataset = NAICSDataset(
    test_df, 
    model_name="answerdotai/ModernBERT-large",
    encoders=shared_encoders
)

# Proceed to Loaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

# Initialize Model and Loss
model = NAICSMultiHeadClassifier(
    num_sectors=24,
    num_subsectors=98,
    num_industries=1066
)

criterion = HierarchicalLoss(alpha=0.2, beta=0.3, gamma=0.5)

# Training Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

# Execution
try:
    # Train the model (uses the Validation set to check progress each epoch)
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=criterion,
        epochs=5,
        lr=2e-5,
        device=device
    )
    
    # Run the final blind test on the Test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*50)
    model.evaluate(test_loader, device=device)
    
    # Save the final model weights
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/naics_multi_head.pth")
    print("\nTraining complete. Model saved to outputs/naics_multi_head.pth")

except Exception as e:
    print(f"An error occurred: {e}")