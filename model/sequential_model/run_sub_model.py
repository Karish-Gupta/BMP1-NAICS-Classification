import torch
from transformers import AutoTokenizer
import pandas as pd
import joblib
from model.sequential_model.sub_model import SequentialSubModel
from model.sequential_model.dataset import NAICSHierarchyDataset

def run_training(step_name, train_csv, val_csv, test_csv, epochs, lr, batch_size):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Raw Data
    print(f"Loading data for {step_name}...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Preprocess (Encode labels, save .pkl, create DataLoaders)
    train_loader, val_loader, num_classes = SequentialSubModel.preprocess(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        step_name=step_name,
        batch_size=batch_size
    )
    
    # Prepare Test Set specifically using the SAVED Label Encoder
    print("Preparing Test Set...")
    le = joblib.load(f"{step_name}_encoder.pkl")
    test_df['label_idx'] = le.transform(test_df['next'].astype(str))
    
    test_ds = NAICSHierarchyDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    model = SequentialSubModel(n_classes=num_classes)
    
    # Run Training
    print(f"Starting training for {step_name}...")
    trained_model = SequentialSubModel.train_full_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name=step_name
    )

    # Final Test Evaluation
    print(f"\n--- Final Evaluation on Test Set [{step_name}] ---")
    
    # Load the best weights saved during training
    model.load_state_dict(torch.load(f"{step_name}_best.pt"))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(ids, mask)
            _, pred = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += pred.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy for {step_name}: {test_acc:.2f}%")
    print(f"Successfully trained, saved, and tested {step_name}!")