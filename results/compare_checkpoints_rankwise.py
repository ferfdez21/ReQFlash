import argparse
import os
import re
import csv
import sys

def parse_metrics(file_path):
    metrics = {
        'grouped_rmsd_below_2_ratio': 0.0,
        'min_rmsd_mean': float('inf')
    }
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()
    
    # grouped_rmsd_below_2_ratio: 0.787 ± 0.152
    m1 = re.search(r'grouped_rmsd_below_2_ratio:\s*([0-9.]+)', content)
    if m1: 
        try:
            metrics['grouped_rmsd_below_2_ratio'] = float(m1.group(1))
        except ValueError:
            pass

    # min_rmsd_range: 2.136 ± 1.804 -> we want the mean (first number)
    # or min_rmsd_mean: 2.136
    m2 = re.search(r'min_rmsd_range:\s*([0-9.]+)', content)
    if m2: 
        try:
            metrics['min_rmsd_mean'] = float(m2.group(1))
        except ValueError:
            pass
            
    return metrics

def get_checkpoints_data(target_dir):
    data = []
    if not os.path.exists(target_dir):
        print(f"Warning: Directory {target_dir} does not exist.")
        return data

    subdirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    for ckpt_dir in subdirs:
        metrics_path = os.path.join(ckpt_dir, "Metrics.txt")
        if os.path.exists(metrics_path):
            m = parse_metrics(metrics_path)
            if m:
                entry = {
                    'name': os.path.basename(ckpt_dir),
                    'path': ckpt_dir
                }
                entry.update(m)
                data.append(entry)
    return data

def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints rank-wise between two models.")
    parser.add_argument("experimental_dir", help="Directory containing experimental model checkpoints")
    parser.add_argument("baseline_dir", help="Directory containing baseline model checkpoints")
    parser.add_argument("--sort-by", default="grouped_rmsd_below_2_ratio", choices=["grouped_rmsd_below_2_ratio", "min_rmsd_mean"], help="Metric to sort by")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending (default is descending for ratio, ascending for rmsd)")
    
    args = parser.parse_args()
    
    # Determine sort order
    if args.sort_by == "grouped_rmsd_below_2_ratio":
        # Default descending (higher is better)
        ascending = args.ascending if args.ascending else False
    else:
        # Default ascending (lower is better for rmsd)
        if args.ascending:
             reverse = False # explicit ascending
             ascending = True
        else:
             # Default behavior for rmsd is Ascending (low is good)
             ascending = True
             
    # Logic for reverse param in sort
    reverse = not ascending

    # Fetch data
    data_experimental = get_checkpoints_data(args.experimental_dir)
    data_baseline = get_checkpoints_data(args.baseline_dir)
    
    if not data_experimental or not data_baseline:
        print("One or both directories contained no valid checkpoints.")
        return

    # Sort
    data_experimental.sort(key=lambda x: x[args.sort_by], reverse=reverse)
    data_baseline.sort(key=lambda x: x[args.sort_by], reverse=reverse)
    
    # Create Table
    print(f"{'Rank':<5} | {'Experimental (Name)':<30} | {'Metric Exp':<10} | {'Baseline (Name)':<30} | {'Metric Base':<10} | {'Diff (E-B)':<10}")
    print("-" * 105)
    
    max_len = max(len(data_experimental), len(data_baseline))
    
    rows = []

    for i in range(max_len):
        row = {}
        row['rank'] = i + 1
        
        # Experimental
        if i < len(data_experimental):
            row['name_exp'] = data_experimental[i]['name']
            row['metric_exp'] = data_experimental[i][args.sort_by]
        else:
            row['name_exp'] = "N/A"
            row['metric_exp'] = None
            
        # Baseline
        if i < len(data_baseline):
            row['name_base'] = data_baseline[i]['name']
            row['metric_base'] = data_baseline[i][args.sort_by]
        else:
            row['name_base'] = "N/A"
            row['metric_base'] = None
            
        # Diff
        if row['metric_exp'] is not None and row['metric_base'] is not None:
            diff = row['metric_exp'] - row['metric_base']
            row['diff'] = diff
            diff_str = f"{diff:+.4f}"
        else:
            row['diff'] = None
            diff_str = "N/A"
        
        m_exp_str = f"{row['metric_exp']:.4f}" if row['metric_exp'] is not None else "N/A"
        m_base_str = f"{row['metric_base']:.4f}" if row['metric_base'] is not None else "N/A"
            
        print(f"{row['rank']:<5} | {row['name_exp']:<30} | {m_exp_str:<10} | {row['name_base']:<30} | {m_base_str:<10} | {diff_str:<10}")
        rows.append(row)
        
    # Save to CSV using built-in csv module
    out_file = "comparison_rankwise.csv"
    try:
        with open(out_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['rank', 'name_exp', 'metric_exp', 'name_base', 'metric_base', 'diff'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved comparison to '{out_file}'")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()
