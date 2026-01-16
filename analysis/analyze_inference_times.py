
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Set base path
BASE_PATH = "/home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis"

# Find all time_records.csv files
# Matches */time_records.csv inside the base path
file_pattern = os.path.join(BASE_PATH, "*_steps", "time_records.csv")
files = glob.glob(file_pattern)

print(f"Found {len(files)} csv files.")

data_frames = []

for f in files:
    # Extract timestep from folder name
    # e.g .../10_steps/time_records.csv -> 10
    folder_name = os.path.basename(os.path.dirname(f))
    try:
        timestep = int(folder_name.split('_')[0])
    except ValueError:
        print(f"Skipping folder {folder_name}, cannot parse timestep.")
        continue
    
    # Read CSV
    # The CSV has extra spaces, so skipinitialspace=True helps.
    try:
        df = pd.read_csv(f, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue
        
    df['timesteps'] = timestep
    data_frames.append(df)

if not data_frames:
    print("No data found!")
    exit()

all_data = pd.concat(data_frames, ignore_index=True)

# Ensure numeric types
all_data['timesteps'] = pd.to_numeric(all_data['timesteps'], errors='coerce')
all_data['length'] = pd.to_numeric(all_data['length'], errors='coerce')

# Sort by timesteps for better plotting
all_data.sort_values(by='timesteps', inplace=True)

metrics = ['total_time', 'sample_time', 'eval_time']

for metric in metrics:
    print(f"Generating graphs for {metric}...")
    
    # Ensure metric is numeric
    all_data[metric] = pd.to_numeric(all_data[metric], errors='coerce')

    # 1. Distribution per Timestep (Violin plot)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='timesteps', y=metric, data=all_data, density_norm='width')
    plt.title(f'Distribution of {metric} per Timestep')
    plt.xlabel('Timesteps')
    plt.ylabel(f'{metric} (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, f"{metric}_distribution.png"))
    plt.close()

    # 2. Distribution per Length (Violin/Box plot)
    # Using a much wider figure to accommodate length 60-128 (approx 70 ticks)
    # If too crowded, consider boxplot or sampling
    plt.figure(figsize=(24, 8))
    # Using boxplot as violinplot might be too noisy/dense for 69 categories
    sns.boxplot(x='length', y=metric, data=all_data, fliersize=1, width=0.7) 
    plt.title(f'Distribution of {metric} per Protein Length')
    plt.xlabel('Protein Length')
    plt.ylabel(f'{metric} (s)')
    plt.grid(True, linestyle='--', alpha=0.3)
    # Improve x-axis labels readability
    plt.xticks(rotation=90, fontsize=8) 
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, f"{metric}_distribution_by_length.png"))
    plt.close()

    # 3. Heatmap
    # Values: Rows=Length, Cols=Timesteps
    pivot_table = all_data.pivot_table(index='length', columns='timesteps', values=metric, aggfunc='mean')

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, cmap="Blues", annot=False, cbar_kws={'label': f'{metric} (s)'})
    plt.title(f'Heatmap of {metric} (Length vs Timesteps)')
    plt.xlabel('Timesteps')
    plt.ylabel('Protein Length')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, f"{metric}_heatmap.png"))
    plt.close()

print("Graphs generated successfully in:", BASE_PATH)
