import pandas as pd
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from utils.feature_extraction import extract_tree_features
from premier_modele.randomtree import rf_model, le
from tqdm import tqdm

# --- 1. DEFINE PATHS & COLUMNS ---
test_folder = "./data/test_data"
# IMPORTANT: These must match exactly what you trained on
feature_cols = ['height', 'volume', 'crown_area', 'point_count'] 

# --- 2. REUSE FEATURE EXTRACTION ---
# We use the same function to ensure consistency
# (Make sure 'process_single_tree' is defined in your current session)
# If not, copy-paste the 'process_single_tree' function from the previous step here.

def extract_test_features():
    print(f"Scanning {test_folder}...")
    files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.laz')]
    
    if not files:
        print("No .laz files found in test folder!")
        return pd.DataFrame()

    print(f"Found {len(files)} test trees. Extracting features...")
    
    # Parallel extraction
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_tree_features, files), total=len(files)))
    
    # Filter out failures (None)
    clean_results = [r for r in results if r is not None]
    
    return pd.DataFrame(clean_results)

# --- 3. RUN EXTRACTION ---
test_df = extract_test_features()

if not test_df.empty:
    print(f"Successfully extracted features for {len(test_df)} trees.")

    # --- 4. PREDICT ---
    # Select only the feature columns (ignore filename for the prediction step)
    X_test = test_df[feature_cols]

    # Predict Class (0, 1, 2...)
    predictions_encoded = rf_model.predict(X_test)

    # Predict Probabilities (Confidence scores)
    # This tells us how sure the model is (e.g., 90% sure it's a Pine)
    probabilities = rf_model.predict_proba(X_test)
    confidence_scores = np.max(probabilities, axis=1)

    # Convert numbers back to names (0 -> "Eucalyptus")
    # We use the LabelEncoder (le) from the training step
    predictions_names = le.inverse_transform(predictions_encoded)

    # --- 5. COMPILE RESULTS ---
    results_df = pd.DataFrame({
        'filename': test_df['filename'],
        'predicted_species': predictions_names,
        'confidence': confidence_scores,
        # We include the features too, just for reference
        'height': test_df['height'],
        'volume': test_df['volume']
    })

    # Sort by low confidence (to see which trees confused the model)
    results_df = results_df.sort_values(by='confidence', ascending=True)

    # --- 6. SAVE & SHOW ---
    print("\n--- PREDICTION RESULTS (Lowest Confidence First) ---")
    print(results_df.head(10))
    
    output_filename = "test_predictions.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Predictions saved to {output_filename}")

else:
    print("Extraction failed or folder was empty.")