import os
import json
import glob

# Base directory
results_dir = "."

# Datasets to merge
target_datasets = ["arxivqa", "docvqa", "tabfquad", "tatdqa"]

def process_directories():
    # Process both normal and segmented directories
    for result_type in ["normal", "segmented"]:
        result_type_dir = os.path.join(results_dir, result_type)
        
        # Skip if directory doesn't exist
        if not os.path.exists(result_type_dir):
            print(f"Directory {result_type_dir} does not exist. Skipping.")
            continue
        
        # Find all model directories
        model_dirs = [d for d in os.listdir(result_type_dir) 
                     if os.path.isdir(os.path.join(result_type_dir, d))]
        
        for model_dir in model_dirs:
            # Special handling for scan models
            if "scan" in model_dir.lower():
                merge_scan_model_results(os.path.join(result_type_dir, model_dir))
            else:
                merge_model_results(os.path.join(result_type_dir, model_dir))

def merge_scan_model_results(model_path):
    model_dir = os.path.basename(model_path)
    print(f"Processing scan model: {model_dir}")
    
    # Create a new merged metrics object for scan model
    merged_metrics = {
        "metadata": {
            "description": f"Merged metrics for scan model {model_dir} on selected datasets",
            "datasets_included": [],
            "timestamp": "",
            "vidore_benchmark_version": "",
            "model_type": "scan"
        },
        "metrics": {}
    }
    
    # Find and process result files for each dataset
    found_datasets = 0
    
    for dataset in target_datasets:
        # For scan models, look deeper in subdirectories if needed
        # First try direct match in the model directory
        pattern = os.path.join(model_path, f"vidore_{dataset}*metrics.json")
        matching_files = glob.glob(pattern)
        
        # If not found directly, try looking in subdirectories
        if not matching_files:
            pattern = os.path.join(model_path, "**", f"vidore_{dataset}*metrics.json")
            matching_files = glob.glob(pattern, recursive=True)
        
        if not matching_files:
            print(f"  No results found for {dataset} in scan model {model_path}")
            continue
        
        # Sort files by name to get consistent results
        matching_files.sort()
        
        # Process the first matching file
        result_file = matching_files[0]
        file_basename = os.path.basename(result_file)
        dataset_name = file_basename.replace('vidore_', '').replace('_metrics.json', '')
        
        print(f"  Including dataset: {dataset_name} from {file_basename}")
        merged_metrics["metadata"]["datasets_included"].append(dataset_name)
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata if present
            if "metadata" in data:
                # Only take version number and timestamp from the first file that has them
                if (not merged_metrics["metadata"]["timestamp"] and 
                    "timestamp" in data["metadata"]):
                    merged_metrics["metadata"]["timestamp"] = data["metadata"]["timestamp"]
                
                if (not merged_metrics["metadata"]["vidore_benchmark_version"] and 
                    "vidore_benchmark_version" in data["metadata"]):
                    merged_metrics["metadata"]["vidore_benchmark_version"] = data["metadata"]["vidore_benchmark_version"]
            
            # For scan models, look for avg_metrics
            if "avg_metrics" in data:
                # Copy average metrics for this dataset
                dataset_key = f"vidore/{dataset_name}"
                merged_metrics["metrics"][dataset_key] = data["avg_metrics"]
                found_datasets += 1
            elif "metrics" in data:
                # Fallback to regular metrics if available
                for dataset_key, metrics in data["metrics"].items():
                    merged_metrics["metrics"][dataset_key] = metrics
                    found_datasets += 1
            else:
                print(f"  No metrics found in {result_file}")
        
        except Exception as e:
            print(f"  Error processing {result_file}: {e}")
    
    # Save the merged metrics
    if found_datasets > 0:
        output_file = os.path.join(model_path, "merged_dataset_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(merged_metrics, f, indent=4)
        print(f"Saved merged metrics for {found_datasets} datasets to {output_file}")
    else:
        print(f"No metrics were merged for scan model {model_dir}")

def merge_model_results(model_path):
    model_dir = os.path.basename(model_path)
    print(f"Processing model: {model_dir}")
    
    # Create a new merged metrics object
    merged_metrics = {
        "metadata": {
            "description": f"Merged metrics for {model_dir} on selected datasets",
            "datasets_included": [],
            "timestamp": "",
            "vidore_benchmark_version": ""
        },
        "metrics": {}
    }
    
    # Find and process result files for each dataset
    found_datasets = 0
    
    for dataset in target_datasets:
        # Look for files matching the dataset pattern
        pattern = os.path.join(model_path, f"vidore_{dataset}*metrics.json")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"  No results found for {dataset} in {model_path}")
            continue
        
        # Sort files by name to get consistent results
        matching_files.sort()
        
        # Process the first matching file
        result_file = matching_files[0]
        file_basename = os.path.basename(result_file)
        dataset_name = file_basename.replace('vidore_', '').replace('_metrics.json', '')
        
        print(f"  Including dataset: {dataset_name} from {file_basename}")
        merged_metrics["metadata"]["datasets_included"].append(dataset_name)
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata if present
            if "metadata" in data:
                # Only take version number and timestamp from the first file that has them
                if (not merged_metrics["metadata"]["timestamp"] and 
                    "timestamp" in data["metadata"]):
                    merged_metrics["metadata"]["timestamp"] = data["metadata"]["timestamp"]
                
                if (not merged_metrics["metadata"]["vidore_benchmark_version"] and 
                    "vidore_benchmark_version" in data["metadata"]):
                    merged_metrics["metadata"]["vidore_benchmark_version"] = data["metadata"]["vidore_benchmark_version"]
            
            # Extract metrics and add to merged metrics
            if "metrics" in data:
                # Copy all metrics for this dataset
                for dataset_key, metrics in data["metrics"].items():
                    merged_metrics["metrics"][dataset_key] = metrics
                    found_datasets += 1
            else:
                print(f"  No metrics found in {result_file}")
        
        except Exception as e:
            print(f"  Error processing {result_file}: {e}")
    
    # Save the merged metrics
    if found_datasets > 0:
        output_file = os.path.join(model_path, "merged_dataset_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(merged_metrics, f, indent=4)
        print(f"Saved merged metrics for {found_datasets} datasets to {output_file}")
    else:
        print(f"No metrics were merged for {model_dir}")

if __name__ == "__main__":
    process_directories()
