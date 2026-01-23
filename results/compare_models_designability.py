import os
import re
import pandas as pd
import argparse

def parse_metrics(file_path):
    metrics = {
        'grouped_rmsd_below_2_ratio': None,
        'min_rmsd_range': None
    }
    if not os.path.exists(file_path):
        return metrics
    with open(file_path, 'r') as f:
        content = f.read()
    
    # grouped_rmsd_below_2_ratio: 0.787 ± 0.152
    m1 = re.search(r'grouped_rmsd_below_2_ratio:\s*(.+)', content)
    if m1: metrics['grouped_rmsd_below_2_ratio'] = m1.group(1).strip()

    # min_rmsd_range: 1.701 ± 1.350
    m2 = re.search(r'min_rmsd_range:\s*(.+)', content)
    if m2: metrics['min_rmsd_range'] = m2.group(1).strip()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare Designability Metrics (Experimental vs Baseline).")
    parser.add_argument("--experimental", required=True, help="Path to experimental model inference output folder")
    parser.add_argument("--baseline", required=True, help="Path to baseline model inference output folder")
    args = parser.parse_args()
    
    data = []
    
    # Experimental
    p1 = os.path.join(args.experimental, "Metrics.txt")
    m1 = parse_metrics(p1)
    data.append({
        'Model': 'Experimental',
        'Path': os.path.basename(os.path.normpath(args.experimental)),
        'grouped_rmsd_below_2_ratio': m1.get('grouped_rmsd_below_2_ratio', 'N/A'),
        'min_rmsd_range': m1.get('min_rmsd_range', 'N/A')
    })
    
    # Baseline
    p2 = os.path.join(args.baseline, "Metrics.txt")
    m2 = parse_metrics(p2)
    data.append({
        'Model': 'Baseline',
        'Path': os.path.basename(os.path.normpath(args.baseline)),
        'grouped_rmsd_below_2_ratio': m2.get('grouped_rmsd_below_2_ratio', 'N/A'),
        'min_rmsd_range': m2.get('min_rmsd_range', 'N/A')
    })
    
    df = pd.DataFrame(data)
    
    # Print table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\nComparison Table:")
    print(df.to_string(index=False))
    
    experimental_name = os.path.basename(os.path.normpath(args.experimental))
    baseline_name = os.path.basename(os.path.normpath(args.baseline))
    outfile = f"results/comparison_metrics_{experimental_name}_vs_{baseline_name}.txt"
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open(outfile, 'w') as f:
        f.write(df.to_string(index=False))
    print(f"\nSaved to {outfile}")

if __name__ == "__main__":
    main()
