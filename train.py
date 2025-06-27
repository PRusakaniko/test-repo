# train.py (Version 2 - Explicit Contract)
import pandas as pd
import joblib
import json
import argparse
from pathlib import Path

# --- The Contract (v2) ---
# This script now receives its input and output paths as command-line arguments.
# This decouples the script from the platform's specific filesystem layout.

# 1. Setup Argument Parser
parser = argparse.ArgumentParser(description="Aethelred Client Training Script")
parser.add_argument("--input-file", type=str, required=True, help="Path to the input CSV data file.")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the directory where outputs will be saved.")
args = parser.parse_args()

# Define output paths based on the provided argument
output_path = Path(args.output_dir)
model_path = output_path / "model.joblib"
results_path = output_path / "results.json"

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# --- Model Training Logic ---
print(f"Executing client training script with explicit contract...")
print(f"Input file: {args.input_file}")
print(f"Output directory: {args.output_dir}")

# 2. Load Data
df = pd.read_csv(args.input_file)
print(f"Loaded {len(df)} records.")

# 3. Feature Engineering & Model Selection
features = ['credit_score', 'income', 'loan_amount', 'age']
target = 'approved'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# 6. Write results to the specified output file
results = {
    "accuracy": accuracy,
    "num_records_trained": len(X_train)
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

# 7. Save Model Artifact to the specified output file
joblib.dump(model, model_path)
print(f"Model artifact saved to {model_path}")

print("Client script execution finished successfully.")
