from model.sequential_model.run_sub_model import run_training

# Configs
step_name = "national_industry"
train_csv = f"data/sequential_model_data/train/naics_step5_national_industry.csv"
val_csv = f"data/sequential_model_data/val/naics_step5_national_industry.csv"
test_csv = f"data/sequential_model_data/test/naics_step5_national_industry.csv"
epochs = 5
lr = 2e-5
batch_size = 32

run_training(step_name=step_name, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv, epochs=epochs, lr=lr, batch_size=batch_size)