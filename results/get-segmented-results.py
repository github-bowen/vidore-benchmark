import os
import json
from sys import stderr
import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Any

# Path to the segmented results
RESULTS_DIR = "./segmented"

# Define all 10 expected datasets
EXPECTED_DATASETS = [
    "arxivqa_test_subsampled",
    "docvqa_test_subsampled",
    "infovqa_test_subsampled",
    "tabfquad_test_subsampled",
    "tatdqa_test",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test"
]
    

def get_model_names() -> List[str]:
    """Get a list of model names from the segmented directory."""
    return [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]

def extract_metrics(model_name: str) -> Dict[str, Dict[str, float]]:
    """Extract metrics from a model's merged_dataset_metrics.json file."""
    metrics_file = os.path.join(RESULTS_DIR, model_name, "merged_dataset_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: No merged_dataset_metrics.json found for {model_name}", file=stderr)
        return {}
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    result = {}
    # Initialize all expected datasets with zeros
    for dataset_name in EXPECTED_DATASETS:
        result[dataset_name] = {
            "ndcg_at_3": 0.0,
            "ndcg_at_5": 0.0,
            "map_at_3": 0.0,
            "map_at_5": 0.0
        }
    
    # Update with actual values from data if they exist
    for dataset_name, dataset_metrics in data.get("metrics", {}).items():
        # Extract the dataset name without the 'vidore/' prefix if present
        clean_dataset_name = dataset_name.replace("vidore/", "")
        
        if clean_dataset_name in result:
            result[clean_dataset_name] = {
                "ndcg_at_3": dataset_metrics.get("ndcg_at_3", 0.0) * 100,
                "ndcg_at_5": dataset_metrics.get("ndcg_at_5", 0.0) * 100,
                "map_at_3": dataset_metrics.get("map_at_3", 0.0) * 100,
                "map_at_5": dataset_metrics.get("map_at_5", 0.0) * 100
            }
    
    return result

def create_metric_table(model_metrics: Dict[str, Dict[str, Dict[str, float]]], metric_name: str) -> pd.DataFrame:
    """Create a DataFrame for a specific metric across all models and datasets."""
    # Use the predefined datasets for consistent ordering
    datasets = EXPECTED_DATASETS
    
    # Create rows for the table
    rows = []
    for model_name, metrics in model_metrics.items():
        row = {'Model': model_name}
        for dataset in datasets:
            if dataset in metrics:
                row[dataset] = metrics[dataset].get(metric_name, 0.0)
            else:
                row[dataset] = 0.0
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Set Model as index and sort columns
    if 'Model' in df.columns:
        df = df.set_index('Model')
    
    return df

def main():
    # Get all model names
    model_names = get_model_names()
    
    # Extract metrics for each model
    model_metrics = {}
    for model_name in model_names:
        model_metrics[model_name] = extract_metrics(model_name)
    
    # Metrics to extract
    metrics = ["ndcg_at_3", "ndcg_at_5", "map_at_3", "map_at_5"]
    
    # Create and display tables for each metric
    for metric in metrics:
        print(f"\n{metric.upper()} Table:")
        df = create_metric_table(model_metrics, metric)
        
        # Rename columns to more readable names
        readable_names = {
            "arxivqa_test_subsampled": "ArxivQ",
            "docvqa_test_subsampled": "DocQ",
            "infovqa_test_subsampled": "InfoQ",
            "tabfquad_test_subsampled": "TabF",
            "tatdqa_test": "TATQ",
            "shiftproject_test": "Shift",
            "syntheticDocQA_artificial_intelligence_test": "AI",
            "syntheticDocQA_energy_test": "Energy",
            "syntheticDocQA_government_reports_test": "Gov.",
            "syntheticDocQA_healthcare_industry_test": "Health"
        }
        df = df.rename(columns=readable_names)
        
        # Format the values to 4 decimal places
        formatted_df = df.map(lambda x: f"{x:.2f}")
        
        # Print table using tabulate for nice formatting
        print(tabulate(formatted_df, headers='keys', tablefmt='pretty'))
        
        # Also save as CSV
        df.to_csv(f"./exp2-results/{metric}_results.csv")
        df.to_markdown(f"./exp2-results/{metric}_results.md")
        print(f"Saved to {metric}_results.csv")
        print(f"Saved to {metric}_results.md")

if __name__ == "__main__":
    main()