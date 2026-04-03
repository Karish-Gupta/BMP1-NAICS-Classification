import torch
import pandas as pd
import joblib
from transformers import AutoTokenizer
from model.sequential_model.sub_model import SequentialSubModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SequentialPipeline:
    def __init__(self, steps, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.models = {}
        self.encoders = {}
        self.steps = steps # e.g., ["step1_sector", "step2_subsector", ...]

        for step in self.steps:
            le = joblib.load(f"{step}_encoder.pkl")
            self.encoders[step] = le
            
            # Initialize and load best weights
            model = SequentialSubModel(n_classes=len(le.classes_))
            model.load_state_dict(torch.load(f"{step}_best.pt", map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[step] = model

def run_full_evaluation(self, test_df):
        """
        Runs the full chain and calculates metrics for every level.
        Expects test_df to have 'inputs' and the original 'NAICS_Code' (6-digit).
        """
        all_preds = {step: [] for step in self.steps}
        all_ground_truth = {step: [] for step in self.steps}
        
        print(f"Processing {len(test_df)} rows through the cascade...")
        
        with torch.no_grad():
            for _, row in test_df.iterrows():
                description = row['inputs']
                full_code = str(row['NAICS_Code'])
                
                current_context = ""
                
                for i, step in enumerate(self.steps):
                    # 1. Prepare input
                    combined_text = f"Context NAICS: {current_context} | {description}"
                    inputs = self.tokenizer(combined_text, return_tensors="pt", 
                                            truncation=True, max_length=512).to(self.device)
                    
                    # 2. Predict
                    outputs = self.models[step](inputs['input_ids'], inputs['attention_mask'])
                    _, pred_idx = outputs.max(1)
                    pred_code = self.encoders[step].inverse_transform([pred_idx.item()])[0]
                    
                    # 3. Store Results
                    all_preds[step].append(pred_code)
                    
                    # Determine target length (2, 3, 4, 5, 6)
                    target_len = i + 2 
                    all_ground_truth[step].append(full_code[:target_len])
                    
                    # Update context for next model in chain
                    current_context = pred_code

        # --- METRICS CALCULATION ---
        report_data = []
        for step in self.steps:
            y_true = all_ground_truth[step]
            y_pred = all_preds[step]
            
            acc = accuracy_score(y_true, y_pred)
            # Using 'weighted' average because NAICS classes are highly imbalanced
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            report_data.append({
                'Step': step,
                'Accuracy': f"{acc:.4f}",
                'Precision': f"{prec:.4f}",
                'Recall': f"{rec:.4f}",
                'F1-Score': f"{f1:.4f}"
            })

        return pd.DataFrame(report_data)

# --- EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ordered_steps = [
        "step1_sector", 
        "step2_subsector", 
        "step3_industry_group", 
        "step4_industry", 
        "step5_national_industry"
    ]

    evaluator = SequentialPipeline(ordered_steps, device)
    
    # Load your test data
    test_df = pd.read_csv("data/split_data_preprocessed/test_data_with_summaries.csv")
    
    # Run Evaluation
    metrics_report = evaluator.run_full_evaluation(test_df)
    
    print("\n" + "="*50)
    print("FINAL CASCADED EVALUATION REPORT")
    print("="*50)
    print(metrics_report.to_string(index=False))