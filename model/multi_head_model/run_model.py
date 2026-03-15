import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.multi_head_model.model import NAICSMultiHeadClassifier
from model.multi_head_model.dataset import NAICSDataset
from model.multi_head_model.loss import HierarchicalLoss

# Load data
train_df = pd.read_csv("data/split_data/train.csv") 
test_df = pd.read_csv("data/split_data/test.csv")

# Initialize Datasets (USES the fitted encoders from train)
train_dataset = NAICSDataset(train_df, model_name="answerdotai/ModernBERT-large")
test_dataset = NAICSDataset(
    test_df, 
    model_name="answerdotai/ModernBERT-large",
    encoders=train_dataset.get_encoders() # Pass the mappings over
)

# Proceed to Loaders and Training
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

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
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=criterion,
        epochs=5,
        lr=2e-5,
        device=device
    )
    
    # Save the final model weights
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/naics_multi_head.pth")
    print("Training complete. Model saved to outputs/")

except Exception as e:
    print(f"An error occurred: {e}")