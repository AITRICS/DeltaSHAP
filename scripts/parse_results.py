# %%
import pandas as pd
import os
import glob
import re
from typing import Dict, List, Tuple
import csv
from pathlib import Path

def find_result_files(base_path: str, data_list: List[str], modeltype_list: List[str], 
                     explainer_list: List[str], seed_list: List[str]) -> List[Tuple[str, str, str, str, str]]:
    """
    Find result files based on the specific pattern provided.
    """
    result_files = []
    
    for modeltype in modeltype_list:
        for dataset in data_list:
            for explainer in explainer_list:
                for seed in seed_list:
                    # Try different possible file paths and extensions
                    # First, look for exact matches
                    base_file_path = f"{base_path}/{modeltype}_{dataset}_{explainer}_{seed}/{dataset}/{modeltype}_{dataset}_{explainer}_{seed}"
                    paths_to_check = [
                        base_file_path,                  # No extension
                        f"{base_file_path}.csv",         # .csv extension
                        f"{base_file_path}/results.csv"  # results.csv in subdirectory
                    ]
                    
                    found = False
                    for path in paths_to_check:
                        if os.path.exists(path):
                            result_files.append((path, modeltype, dataset, explainer, seed))
                            # print(f"Found result file: {path}")
                            found = True
                            break
                    
                    # If not found, try to find files with additional suffixes
                    if not found:
                        # Look for files in the base directory that match the pattern
                        pattern = f"{base_path}/{modeltype}_{dataset}_{explainer}_*_{seed}"
                        matching_dirs = glob.glob(pattern)
                        
                        for dir_path in matching_dirs:
                            dir_name = os.path.basename(dir_path)
                            parts = dir_name.split('_')
                            
                            # Extract the actual explainer name with suffix
                            if len(parts) > 3:  # modeltype_dataset_explainer_suffix_seed
                                actual_explainer = '_'.join(parts[2:-1])  # Join all parts between dataset and seed
                                
                                # Check if this is a variant of the explainer we're looking for
                                if parts[2] == explainer:
                                    result_path = f"{dir_path}/{dataset}/{dir_name}"
                                    paths_to_check = [
                                        result_path,
                                        f"{result_path}.csv",
                                        f"{result_path}/results.csv"
                                    ]
                                    
                                    for path in paths_to_check:
                                        if os.path.exists(path):
                                            result_files.append((path, modeltype, dataset, actual_explainer, seed))
                                            # print(f"Found result file with suffix: {path}")
                                            found = True
                                            break
                                    
                                    if found:
                                        break
    
    # print(f"Found {len(result_files)} result files")
    return result_files

def parse_results(file_info_list: List[Tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """
    Parse the result files and combine them into a single DataFrame.
    """
    all_results = []
    explainer_map = {
        'LIME': 'lime',
        'SHAP': 'shap',
        'KERNELSHAP': 'kernelshap',
        'GRADIENTSHAP': 'gradientshap',
        'IG': 'ig',
        'DEEPLIFT': 'deeplift',
        'FO': 'fo',
        'AFO': 'afo',
        'FIT': 'fit',
        'FIT_GENERATOR': 'fit',
        'WINIT': 'winit',
        'DELTASHAP': 'deltashap',
        'TIMING': 'timing'
    }
    
    for file_path, modeltype, dataset, explainer, seed in file_info_list:
        try:
            # print(f"Reading file: {file_path}")
            
            # Check if file exists and has content
            if not os.path.exists(file_path):
                # print(f"Warning: File {file_path} does not exist")
                continue
                
            file_size = os.path.getsize(file_path)
            # print(f"File size: {file_size} bytes")
            if file_size == 0:
                # print(f"Warning: File {file_path} is empty")
                continue
            
            # Try different read methods based on file extension
            if file_path.endswith('.csv'):
                # Try with different delimiters
                try:
                    df = pd.read_csv(file_path)
                except:
                    try:
                        df = pd.read_csv(file_path, delimiter='\t')
                    except:
                        df = pd.read_csv(file_path, delimiter=',', engine='python')
            else:
                # Try to read as directory with multiple files
                try:
                    if os.path.isdir(file_path):
                        csv_files = glob.glob(f"{file_path}/*.csv")
                        if csv_files:
                            df = pd.read_csv(csv_files[0])
                        else:
                            # print(f"No CSV files found in directory {file_path}")
                            continue
                    else:
                        # Try to read without extension
                        df = pd.read_csv(file_path, header=None)
                        # If successful but no column names, add default column names
                        if all(isinstance(col, int) for col in df.columns):
                            df.columns = [f"col_{i}" for i in range(len(df.columns))]
                except Exception as e:
                    # print(f"Failed to read file {file_path}: {str(e)}")
                    continue
            
            # # Print the first few rows to debug
            # print(f"File content preview: {df.head()}")
            # print(f"Columns: {df.columns.tolist()}")
            # print(f"Shape: {df.shape}")            
            # print(f"\n\n\n\n\n{df=}\n\n\n\n\n")
            
            # Check if the first row contains column headers
            if df.shape[0] > 0 and all(col.startswith('col_') for col in df.columns):
                # If the first row looks like headers, use it as column names
                if 'dataset' in df.iloc[0].values and 'explainer' in df.iloc[0].values:
                    new_headers = df.iloc[0].values
                    df = df.iloc[1:].reset_index(drop=True)
                    df.columns = new_headers
                    # print(f"Used first row as headers: {df.columns.tolist()}")
            
            # Add metadata columns if they don't exist
            if "dataset" not in df.columns:
                df["dataset"] = dataset
            if "modeltype" not in df.columns:
                df["modeltype"] = modeltype
            if "explainer" not in df.columns:
                df["explainer"] = explainer.lower()
            else:
                # Map to lowercase and handle special cases
                df["explainer"] = df["explainer"].map(lambda x: explainer_map.get(str(x).upper(), str(x).lower()))
            if "seed" not in df.columns:
                df["seed"] = seed
                
            # Check if we have any data
            if df.empty:
                print(f"Warning: DataFrame is empty after reading {file_path}")
                continue
            
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # print(f"{all_results=}")
    if not all_results:
        print("No results could be parsed from any files.")
        return pd.DataFrame()
    
    try:
        combined_results = pd.concat(all_results, ignore_index=True)
        print(f"Combined results shape: {combined_results.shape}")
        print(f"Combined results columns: {combined_results.columns.tolist()}")
        return combined_results
    except Exception as e:
        print(f"Error combining results: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def generate_model_data_summaries(results_df, modeltype_list, data_list, explainer_list):
    """
    Generate summary tables for each model and dataset combination.
    """
    if any(col not in results_df.columns for col in ["dataset", "modeltype", "explainer"]):
        print("Warning: Required columns 'dataset', 'modeltype', and 'explainer' not found in results.")
        return
    
    # Define the explainer order and names
    explainer_config = {
        "lime": "LIME",
        "shap": "SHAP",
        "kernelshap": "KernelSHAP",
        "gradientshap": "GradientSHAP",
        "ig": "IG",
        "deeplift": "DeepLIFT",
        "fo": "FO",
        "afo": "AFO",
        "fit": "FIT",
        "winit": "WinIT",
        "deltashap": "DeltaSHAP",
        "timing": "Timing"
    }
    
    # Define metrics to include in the summary
    metrics_config = [
        "AUPD", "AUAUCD", "AUAPRD", "AUF1D",
        "APPP", "AUAUCP", "AUAPRP", "AUF1P",
        "elapsed_time"
    ]
    
    # Process each model and dataset combination
    for modeltype in modeltype_list:
        for dataset in data_list:
            print(f"\nProcessing model: {modeltype}, dataset: {dataset}")
            
            # Filter data for this model and dataset
            filtered_df = results_df[(results_df["modeltype"] == modeltype) & 
                                    (results_df["dataset"] == dataset)]
            
            # Prepare summary data
            summary_data = []
            
            # Create a row for each explainer in the ordered list
            for explainer_base in explainer_list:
                # Skip explainers not in config
                if explainer_base not in explainer_config:
                    print(f"Warning: Explainer '{explainer_base}' not found in explainer_config")
                    continue
                
                # Create row with explainer name
                row = {
                    "Name": explainer_config[explainer_base]
                }
                
                # Get data for this explainer, including variants with suffixes
                explainer_data = filtered_df[filtered_df["explainer"].str.startswith(explainer_base) 
                                           if not filtered_df.empty and "explainer" in filtered_df.columns 
                                           else False]
                
                print(f"{modeltype=}, {dataset=}, {explainer_base=}, {explainer_data=}")
                
                # If we have data for this explainer, calculate statistics
                if not explainer_data.empty:
                    for metric in metrics_config:
                        if metric in explainer_data.columns:
                            try:
                                if metric != "elapsed_time" and metric != "avg_masked_count":
                                    mean_val = explainer_data[metric].astype(float).mean()
                                    se_val = explainer_data[metric].astype(float).std(ddof=1) / (len(explainer_data) ** 0.5) if len(explainer_data) > 0 else 0
                                    row[metric] = f"{mean_val * 10000:.2f}±{se_val * 10000:.2f}"
                                else:
                                    mean_val = explainer_data[metric].astype(float).mean()
                                    se_val = explainer_data[metric].astype(float).std(ddof=1) / (len(explainer_data) ** 0.5) if len(explainer_data) > 0 else 0
                                    row[metric] = f"{mean_val:.2f}±{se_val:.2f}"
                            except Exception as e:
                                print(f"Error calculating statistics for {metric}: {str(e)}")
                                row[metric] = "N/A"
                        else:
                            row[metric] = "N/A"
                else:
                    for metric in metrics_config:
                        row[metric] = "N/A"
                
                summary_data.append(row)
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Ensure all columns exist
            for col in ["Name"] + metrics_config:
                if col not in summary_df.columns:
                    summary_df[col] = "N/A"
            
            # Select and order columns
            summary_df = summary_df[["Name"] + metrics_config]
            
            print(f"{summary_df=}")
            
            # Save summary
            filename = f"{modeltype}_{dataset}_summary.csv"
            summary_df.to_csv(filename, index=False)
            print(f"Saved summary to {filename}")

def main():
    # Define parameters
    data_list = ["mimic3o", "physionet19"]
    # modeltype_list = ["LSTM", "SEFT"]
    modeltype_list = ["LSTM"]
    explainer_list = ["deltashap", "ig", "deeplift", "fo", "afo", "fit", "winit", "lime", "kernelshap", "gradientshap"]
    seed_list = ["0", "1", "2", "3", "4"]
    base_path = "../output/neurips"
    
    try:
        print(f"Finding result files in {base_path}...")
        file_info_list = find_result_files(base_path, data_list, modeltype_list, explainer_list, seed_list)
        
        if not file_info_list:
            print("No result files found. Check the paths and file naming pattern.")
            return
        
        print(f"Found {len(file_info_list)} result files")
        
        results_df = parse_results(file_info_list)
        
        if results_df.empty:
            print("No results could be parsed from the files.")
            return
        
        # Generate summaries for each model-dataset combination
        generate_model_data_summaries(results_df, modeltype_list, data_list, explainer_list)
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






# # %%
# import pandas as pd
# import os
# import glob
# import re
# from typing import Dict, List, Tuple
# import csv
# from pathlib import Path

# def find_ablation_result_files(base_path: str, modeltype: str, data: str, explainer: str,
#                               seed_list: List[str], n_samples_list: List[str], 
#                               normalize_list: List[str], baseline_list: List[str]) -> List[Tuple[str, Dict]]:
#     """
#     Find result files from DeltaSHAP ablation study with parameters varied.
#     """
#     result_files = []
    
#     # Check for standard settings first
#     standard_n_samples = "25"
#     standard_normalize = "True"
#     standard_baseline = "carryforward"
    
#     # Process each seed
#     for seed in seed_list:
#         # Standard configuration
#         file_pattern = f"{modeltype}_{data}_{explainer}_{seed}_{standard_n_samples}_{standard_normalize}_{standard_baseline}"
#         file_path = f"{base_path}/{file_pattern}/{data}/{file_pattern}"
        
#         print(f"{file_path=}")
#         print(f"{os.path.exists(file_path)=}")

#         if os.path.exists(file_path) or os.path.exists(f"{file_path}.csv"):
#             path_to_use = file_path if os.path.exists(file_path) else f"{file_path}.csv"
#             params = {
#                 "modeltype": modeltype,
#                 "dataset": data,
#                 "explainer": explainer,
#                 "seed": seed,
#                 "n_samples": standard_n_samples,
#                 "normalize": standard_normalize,
#                 "baseline": standard_baseline,
#                 "ablation_type": "standard"
#             }
#             result_files.append((path_to_use, params))
#             print(f"Found standard file: {path_to_use}")
    
#     # Process n_samples ablation
#     for n_samples in n_samples_list:
#         if n_samples == standard_n_samples:
#             continue  # Skip the standard setting as it's already processed

#         for seed in seed_list:
#             file_pattern = f"{modeltype}_{data}_{explainer}_{seed}_{n_samples}_{standard_normalize}_{standard_baseline}"
#             file_path = f"{base_path}/{file_pattern}/{data}/{file_pattern}"

#             print(f"{file_path=}")
#             print(f"{os.path.exists(file_path)=}")

#             if os.path.exists(file_path) or os.path.exists(f"{file_path}.csv"):
#                 path_to_use = file_path if os.path.exists(file_path) else f"{file_path}.csv"
#                 params = {
#                     "modeltype": modeltype,
#                     "dataset": data,
#                     "explainer": explainer,
#                     "seed": seed,
#                     "n_samples": n_samples,
#                     "normalize": standard_normalize,
#                     "baseline": standard_baseline,
#                     "ablation_type": "n_samples"
#                 }
#                 result_files.append((path_to_use, params))
#                 print(f"Found n_samples ablation file: {path_to_use}")
    
#     # Process baseline ablation
#     for baseline in baseline_list:
#         if baseline == standard_baseline:
#             continue  # Skip the standard setting
            
#         for seed in seed_list:
#             file_pattern = f"{modeltype}_{data}_{explainer}_{seed}_{standard_n_samples}_{standard_normalize}_{baseline}"
#             file_path = f"{base_path}/{file_pattern}/{data}/{file_pattern}"

#             print(f"{file_path=}")
#             print(f"{os.path.exists(file_path)=}")

#             if os.path.exists(file_path) or os.path.exists(f"{file_path}.csv"):
#                 path_to_use = file_path if os.path.exists(file_path) else f"{file_path}.csv"
#                 params = {
#                     "modeltype": modeltype,
#                     "dataset": data,
#                     "explainer": explainer,
#                     "seed": seed,
#                     "n_samples": standard_n_samples,
#                     "normalize": standard_normalize,
#                     "baseline": baseline,
#                     "ablation_type": "baseline"
#                 }
#                 result_files.append((path_to_use, params))
#                 print(f"Found baseline ablation file: {path_to_use}")
    
#     # Process normalize ablation
#     for normalize in normalize_list:
#         if normalize == standard_normalize:
#             continue  # Skip the standard setting
            
#         for seed in seed_list:
#             file_pattern = f"{modeltype}_{data}_{explainer}_{seed}_{standard_n_samples}_{normalize}_{standard_baseline}"
#             file_path = f"{base_path}/{file_pattern}/{data}/{file_pattern}"

#             print(f"{file_path=}")
#             print(f"{os.path.exists(file_path)=}")

#             if os.path.exists(file_path) or os.path.exists(f"{file_path}.csv"):
#                 path_to_use = file_path if os.path.exists(file_path) else f"{file_path}.csv"
#                 params = {
#                     "modeltype": modeltype,
#                     "dataset": data,
#                     "explainer": explainer,
#                     "seed": seed,
#                     "n_samples": standard_n_samples,
#                     "normalize": normalize,
#                     "baseline": standard_baseline,
#                     "ablation_type": "normalize"
#                 }
#                 result_files.append((path_to_use, params))
#                 print(f"Found normalize ablation file: {path_to_use}")
    
#     return result_files

# def parse_ablation_results(file_info_list: List[Tuple[str, Dict]]) -> pd.DataFrame:
#     """
#     Parse the ablation study result files and combine them into a single DataFrame.
#     """
#     all_results = []
    
#     for file_path, params in file_info_list:
#         try:
#             df = pd.read_csv(file_path)
            
#             # Add parameters as columns
#             for key, value in params.items():
#                 df[key] = value
            
#             all_results.append(df)
            
#         except Exception as e:
#             print(f"Error processing {file_path}: {str(e)}")
#             import traceback
#             traceback.print_exc()
    
#     if not all_results:
#         return pd.DataFrame()
    
#     try:
#         combined_results = pd.concat(all_results, ignore_index=True)
#         return combined_results
#     except Exception as e:
#         print(f"Error combining results: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return pd.DataFrame()

# def generate_ablation_summaries(results_df: pd.DataFrame, metrics_list: List[str]):
#     """
#     Generate summary tables for each ablation type.
#     """
#     # Check if we have the necessary columns
#     required_cols = ["ablation_type", "n_samples", "normalize", "baseline", "seed"]
#     if any(col not in results_df.columns for col in required_cols):
#         print(f"Warning: Required columns {required_cols} not found in results.")
#         missing = [col for col in required_cols if col not in results_df.columns]
#         print(f"Missing columns: {missing}")
#         return
    
#     # Group by ablation type
#     for ablation_type in results_df["ablation_type"].unique():
#         ablation_df = results_df[results_df["ablation_type"] == ablation_type]
        
#         if ablation_df.empty:
#             print(f"No data found for ablation type: {ablation_type}")
#             continue
        
#         print(f"\nProcessing ablation type: {ablation_type}")
        
#         # Create summary based on ablation type
#         if ablation_type == "n_samples":
#             param_values = ablation_df["n_samples"].unique()
#             param_name = "n_samples"
#         elif ablation_type == "baseline":
#             param_values = ablation_df["baseline"].unique()
#             param_name = "baseline"
#         elif ablation_type == "normalize":
#             param_values = ablation_df["normalize"].unique()
#             param_name = "normalize"
#         elif ablation_type == "standard":
#             # Just use a placeholder for the standard configuration
#             param_values = ["standard"]
#             param_name = "config"
#         else:
#             print(f"Unknown ablation type: {ablation_type}")
#             continue
        
#         # Create summary data
#         summary_data = []
        
#         for param_value in param_values:
#             if ablation_type == "standard":
#                 filtered_df = ablation_df  # All standard data
#             else:
#                 filtered_df = ablation_df[ablation_df[param_name] == param_value]
            
#             row = {
#                 "Configuration": f"{param_name}={param_value}"
#             }
            
#             # Calculate statistics for each metric
#             for metric in metrics_list:
#                 if metric in filtered_df.columns:
#                     mean_val = filtered_df[metric].mean()
#                     se_val = filtered_df[metric].std(ddof=1) / (len(filtered_df) ** 0.5) if len(filtered_df) > 0 else 0
                    
#                     # Format based on metric type
#                     if metric in ["Wall-Clock Time", "avg_masked_count"]:
#                         row[metric] = f"{mean_val:.2f}±{se_val:.2f}"
#                     else:
#                         row[metric] = f"{mean_val * 10000:.2f}±{se_val * 10000:.2f}"
#                 else:
#                     row[metric] = "N/A"
            
#             summary_data.append(row)
        
#         # Create DataFrame
#         summary_df = pd.DataFrame(summary_data)
        
#         # Ensure all columns exist
#         for col in ["Configuration"] + metrics_list:
#             if col not in summary_df.columns:
#                 summary_df[col] = "N/A"
        
#         # Select and order columns
#         summary_df = summary_df[["Configuration"] + metrics_list]
        
#         # Save summary
#         modeltype = ablation_df["modeltype"].iloc[0]
#         dataset = ablation_df["dataset"].iloc[0]
#         explainer = ablation_df["explainer"].iloc[0]
        
#         filename = f"{modeltype}_{dataset}_{explainer}_{ablation_type}_ablation_summary.csv"
#         summary_df.to_csv(filename, index=False)
#         print(f"Saved summary to {filename}")

# def plot_ablation_results(results_df: pd.DataFrame, metrics_list: List[str], output_dir: str = "plots"):
#     """
#     Generate plots for ablation studies.
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process each ablation type
#     for ablation_type in results_df["ablation_type"].unique():
#         if ablation_type == "standard":
#             continue  # Skip standard configuration for plotting
            
#         ablation_df = results_df[results_df["ablation_type"] == ablation_type]
        
#         if ablation_df.empty:
#             continue
        
#         # Set parameter name based on ablation type
#         if ablation_type == "n_samples":
#             param_name = "n_samples"
#             # Convert to numeric for proper ordering in plots
#             ablation_df[param_name] = pd.to_numeric(ablation_df[param_name])
#         elif ablation_type == "baseline":
#             param_name = "baseline"
#         elif ablation_type == "normalize":
#             param_name = "normalize"
#         else:
#             continue
        
#         modeltype = ablation_df["modeltype"].iloc[0]
#         dataset = ablation_df["dataset"].iloc[0]
#         explainer = ablation_df["explainer"].iloc[0]
        
#         # Plot each metric
#         for metric in metrics_list:
#             if metric not in ablation_df.columns:
#                 continue
                
#             plt.figure(figsize=(10, 6))
            
#             # Create boxplot
#             sns.boxplot(x=param_name, y=metric, data=ablation_df)
            
#             # Add individual points
#             sns.stripplot(x=param_name, y=metric, data=ablation_df, 
#                          color='black', size=5, alpha=0.5)
            
#             # Add mean value as text
#             for i, param_val in enumerate(sorted(ablation_df[param_name].unique())):
#                 mean_val = ablation_df[ablation_df[param_name] == param_val][metric].mean()
#                 plt.text(i, ablation_df[metric].max() * 1.05, 
#                         f"{mean_val:.4f}", ha='center')
            
#             plt.title(f"{metric} vs {param_name} for {explainer}\n({modeltype}, {dataset})")
#             plt.tight_layout()
            
#             # Save plot
#             filename = f"{output_dir}/{modeltype}_{dataset}_{explainer}_{ablation_type}_{metric}.png"
#             plt.savefig(filename, dpi=300)
#             plt.close()
#             print(f"Saved plot to {filename}")

# def main():
#     # Define parameters
#     data_list = ["mimic3o"]  # You can extend this list as needed
#     modeltype_list = ["LSTM"]  # You can extend this list as needed
#     explainer_list = ["deltashap"]
#     seed_list = ["0", "1", "2", "3", "4"]
    
#     # DeltaSHAP specific parameters from the script
#     n_samples_list = ["1", "10", "25", "100"]
#     normalize_list = ["True", "False"]
#     baseline_list = ["carryforward", "zero"]
    
#     base_path = "../output"  # Base path for output files
    
#     # Define metrics to analyze
#     metrics_list = [
#         "AUPD", "AUAUCD", "AUAPRD", "AUF1D",
#         "APPP", "AUAUCP", "AUAPRP", "AUF1P",
#         "Wall-Clock Time"
#     ]
    
#     try:
#         all_results = []
        
#         # Process each model and dataset combination
#         for modeltype in modeltype_list:
#             for data in data_list:
#                 print(f"\nProcessing modeltype: {modeltype}, dataset: {data}")
                
#                 # Find result files for ablation study
#                 file_info_list = find_ablation_result_files(
#                     base_path, 
#                     modeltype, 
#                     data, 
#                     "deltashap",  # We're focusing on deltashap for ablation
#                     seed_list,
#                     n_samples_list,
#                     normalize_list,
#                     baseline_list
#                 )
                
#                 if not file_info_list:
#                     print(f"No result files found for {modeltype} on {data}")
#                     continue
                
#                 # Parse results
#                 results_df = parse_ablation_results(file_info_list)
                
#                 if results_df.empty:
#                     print(f"No results could be parsed for {modeltype} on {data}")
#                     continue
                
#                 # Generate summaries
#                 generate_ablation_summaries(results_df, metrics_list)
                
#                 # Generate plots
#                 # plot_ablation_results(results_df, metrics_list)
                
#                 # Add to overall results
#                 all_results.append(results_df)
        
#         # Combine all results if we have multiple models/datasets
#         if len(all_results) > 1:
#             combined_df = pd.concat(all_results, ignore_index=True)
#             combined_df.to_csv("all_ablation_results.csv", index=False)
#             print("Saved combined results to all_ablation_results.csv")
    
#     except Exception as e:
#         print(f"Error in main execution: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()
# %%
