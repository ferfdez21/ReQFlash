import argparse
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_metrics(file_path):
    metrics = {
        'grouped_rmsd_below_2_ratio': None,
        'min_rmsd_range': None
    }
    if not os.path.exists(file_path):
        return metrics
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the mean value (float) from "Value ± Std" format
    # grouped_rmsd_below_2_ratio: 0.787 ± 0.152
    m1 = re.search(r'grouped_rmsd_below_2_ratio:\s*([0-9.]+)', content)
    if m1: 
        try:
            metrics['grouped_rmsd_below_2_ratio'] = float(m1.group(1))
        except ValueError:
            pass

    # min_rmsd_range: 1.701 ± 1.350
    m2 = re.search(r'min_rmsd_range:\s*([0-9.]+)', content)
    if m2: 
        try:
            metrics['min_rmsd_range'] = float(m2.group(1))
        except ValueError:
            pass
            
    return metrics

def collect_data(target_dir, model_name, multi_ckpt):
    data = []
    if not os.path.exists(target_dir):
        print(f"Warning: Directory {target_dir} does not exist.")
        return data

    if multi_ckpt:
        # Iterate over subdirectories (checkpoints)
        print(f"Multi-ckpt enabled: Scanning subdirectories in {target_dir} for {model_name}...")
        subdirs = sorted([os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
        print(f"Found {len(subdirs)} subdirectories.")
        
        for ckpt_dir in subdirs:
            metrics_path = os.path.join(ckpt_dir, "Metrics.txt")
            if os.path.exists(metrics_path):
                m = parse_metrics(metrics_path)
                if m['grouped_rmsd_below_2_ratio'] is not None and m['min_rmsd_range'] is not None:
                    data.append({
                        'Model': model_name,
                        'Source': os.path.basename(ckpt_dir), # Checkpoint name
                        'grouped_rmsd_below_2_ratio': m['grouped_rmsd_below_2_ratio'],
                        'min_rmsd_range': m['min_rmsd_range']
                    })
    else:
        # Single checkpoint mode: Read CSV to generate distribution over lengths
        print(f"Single-ckpt mode: Reading CSV in {target_dir} for {model_name}...")
        csv_path = os.path.join(target_dir, "All_Results_Origin.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'length' in df.columns and 'min_rmsd' in df.columns:
                    grouped = df.groupby('length')
                    for length, group in grouped:
                        # logical equivalent to 'grouped_rmsd_below_2_ratio' component calculation
                        designability = (group['min_rmsd'] < 2).mean()
                        
                        # logical equivalent to 'min_rmsd_range' component (Average min_rmsd)
                        avg_min_rmsd = group['min_rmsd'].mean()
                        
                        data.append({
                            'Model': model_name,
                            'Source': f"Length {length}",
                            'grouped_rmsd_below_2_ratio': designability,
                            'min_rmsd_range': avg_min_rmsd
                        })
                else:
                    print(f"Warning: Required columns missing in {csv_path}")
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")
        else:
            print(f"CSV file not found: {csv_path}")
            
    return data

def main():
    parser = argparse.ArgumentParser(description="Boxplot Designability Metrics Distribution (Experimental vs Baseline).")
    parser.add_argument("--experimental_dir", required=True, help="Directory containing experimental model checkpoints")
    parser.add_argument("--baseline_dir", required=True, help="Directory containing baseline model checkpoints")
    parser.add_argument("--multi-ckpt", action="store_true", help="If Set, looks for checkpoints INSIDE the given directories. If Not Set, treats directories AS the checkpoints.")
    args = parser.parse_args()
    
    # Collect data
    data_experimental = collect_data(args.experimental_dir, "Experimental", args.multi_ckpt)
    data_baseline = collect_data(args.baseline_dir, "Baseline", args.multi_ckpt)
    
    all_data = data_experimental + data_baseline
    
    if not all_data:
        print("No valid data found.")
        return

    df = pd.DataFrame(all_data)
    
    if not os.path.exists('results'):
        os.makedirs('results')

    sns.set(style="whitegrid")

    # Plot 1: grouped_rmsd_below_2_ratio
    plt.figure(figsize=(8, 6))
    # If single point, boxplot is weird, but it works.
    sns.boxplot(x='Model', y='grouped_rmsd_below_2_ratio', data=df, width=0.5, palette="Set2")
    sns.stripplot(x='Model', y='grouped_rmsd_below_2_ratio', data=df, color='black', alpha=0.5, jitter=True) # Add points
    plt.title("grouped_rmsd_below_2_ratio", fontsize=16)
    plt.ylabel("Ratio", fontsize=12)
    plt.xlabel("")
    
    out1 = "results/distribution_grouped_rmsd_below_2_ratio.png"
    plt.savefig(out1, dpi=300)
    print(f"Saved plot to {out1}")
    plt.close()

    # Plot 2: min_rmsd_range
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='min_rmsd_range', data=df, width=0.5, palette="Set2")
    sns.stripplot(x='Model', y='min_rmsd_range', data=df, color='black', alpha=0.5, jitter=True) # Add points
    plt.title("min_rmsd_range", fontsize=16)
    plt.ylabel("RMSD value (mean)", fontsize=12)
    plt.xlabel("")

    out2 = "results/distribution_min_rmsd_range.png"
    plt.savefig(out2, dpi=300)
    print(f"Saved plot to {out2}")
    plt.close()
    
    print("\nSummary Statistics:")
    print(df.groupby('Model')[['grouped_rmsd_below_2_ratio', 'min_rmsd_range']].agg(['mean', 'std', 'count']))

if __name__ == "__main__":
    main()
