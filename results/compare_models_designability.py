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
    parser = argparse.ArgumentParser(description="Compare Designability Metrics (QFlash vs QFlow).")
    parser.add_argument("reqflash_path", type=str, help="Path to ReQFlash inference outputs folder (e.g. .../epoch102)")
    args = parser.parse_args()
    
    timesteps = [10, 20, 50, 100, 200, 300, 400, 500]
    
    reqflash_base = args.reqflash_path
    # Baseline path
    reqflow_base = "inference_outputs/ckpts/qflow_scope"
    
    data = []
    
    for t in timesteps:
        # QFlash
        p1 = os.path.join(reqflash_base, "Metrics.txt")
        m1 = parse_metrics(p1)
        data.append({
            'Model': 'QFlash',
            'Step': t,
            'grouped_rmsd_below_2_ratio': m1['grouped_rmsd_below_2_ratio'],
            'min_rmsd_range': m1['min_rmsd_range']
        })
        
        # QFlow (Baseline)
        p2 = os.path.join(reqflow_base, f"{t}_steps", "Metrics.txt")
        m2 = parse_metrics(p2)
        data.append({
            'Model': 'QFlow',
            'Step': t,
            'grouped_rmsd_below_2_ratio': m2['grouped_rmsd_below_2_ratio'],
            'min_rmsd_range': m2['min_rmsd_range']
        })
        
    df = pd.DataFrame(data)
    
    # Sort: QFlow first, then QFlash? Or just by Model name?
    # User said: "rows per model (QFlow (baseline) and QFlash), then per timestep"
    # So model primary sort key.
    # Let's sort QFlow first.
    # We can use a custom sort or just descending 'Model' since 'QFlow' > 'QFlash'.
    df = df.sort_values(by=['Model', 'Step'], ascending=[False, False]) 
    
    # Print table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\nComparison Table:")
    print(df.to_string(index=False))
    
    # Extract epoch for filename
    # matches epoch102 or epoch=102
    match = re.search(r'epoch?(\d+)', reqflash_base)
    epoch = match.group(1) if match else "unknown"
    
    outfile = f"results/designability_comparison_epoch{epoch}.txt"
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open(outfile, 'w') as f:
        f.write(df.to_string(index=False))
    print(f"\nSaved to {outfile}")

if __name__ == "__main__":
    main()
