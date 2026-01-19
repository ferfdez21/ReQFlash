import pandas as pd
import argparse
import os
import re
import numpy as np

def compute_metrics_binned(csv_path, num_bins=8):
    # Reads All_Results_Origin.csv and bins by length
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if 'length' not in df.columns or 'min_rmsd' not in df.columns:
        print(f"Warning: columns missing in {csv_path}")
        return pd.DataFrame()
        
    # Dynamic Binning
    # Infer range from data or assume standard 60-128 if within range
    # To be safe and consistent across both datasets (which might have missing lengths),
    # let's use the explicit global range 60-128 which the user mentioned.
    min_len = 60
    max_len = 129 # 128 is last length, so 129 for right=False
    
    # Check if data is outside this range?
    dmin = df['length'].min()
    dmax = df['length'].max()
    if dmin < min_len: min_len = dmin
    if dmax >= max_len: max_len = dmax + 1
    
    edges = np.linspace(min_len, max_len, num_bins + 1).astype(int)
    # Create labels like "60-68"
    # edges[i] is inclusive start, edges[i+1] is exclusive end
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges)-1)]
    
    df['bin'] = pd.cut(df['length'], bins=edges, labels=labels, right=False)
    
    results = []
    grouped = df.groupby('bin', observed=False) 
    
    for bin_name, group in grouped:
        if group.empty:
            designability = 0.0
            mean_rmsd = 0.0
            std_rmsd = 0.0
        else:
            designability = (group['min_rmsd'] < 2).mean()
            mean_rmsd = group['min_rmsd'].mean()
            std_rmsd = group['min_rmsd'].std()
        
        results.append({
            'length_bin': bin_name,
            'designability': designability,
            'scRMSD_str': f"{mean_rmsd:.3f} ± {std_rmsd:.3f}"
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Compare Designability by Length Bins (500 steps).")
    parser.add_argument("reqflash_path", help="Path to ReQFlash epoch folder (e.g. .../epoch102)")
    args = parser.parse_args()
    
    flash_csv = os.path.join(args.reqflash_path, "All_Results_Origin.csv")
    flow_csv = "inference_outputs/ckpts/qflow_scope/500_steps/All_Results_Origin.csv"
    num_bins = 8
    
    print(f"Reading QFlash: {flash_csv}...")
    df_flash = compute_metrics_binned(flash_csv, num_bins=num_bins)
    
    print(f"Reading QFlow: {flow_csv}...")
    df_flow = compute_metrics_binned(flow_csv, num_bins=num_bins)
    
    if df_flash.empty or df_flow.empty:
        print("Data extraction failed.")
        return

    # Merge
    df_flash = df_flash.rename(columns={'designability': 'QFlash_Designability', 'scRMSD_str': 'QFlash_scRMSD'})
    df_flow = df_flow.rename(columns={'designability': 'QFlow_Designability', 'scRMSD_str': 'QFlow_scRMSD'})
    
    merged = pd.merge(df_flow, df_flash, on='length_bin', how='outer')
    # merged = merged.sort_values(by='length_bin') # Already sorted by categorical
    
    cols = ['length_bin', 'QFlow_Designability', 'QFlow_scRMSD', 'QFlash_Designability', 'QFlash_scRMSD']
    merged = merged[cols]
    
    # Sort bins numerically by lower bound
    # Extract lower bound from "60-77" -> 60 using regex
    merged['sort_key'] = merged['length_bin'].astype(str).str.extract(r'^(\d+)').astype(int)
    merged = merged.sort_values(by='sort_key')
    merged = merged.drop(columns=['sort_key'])
    
    merged['QFlow_Designability'] = merged['QFlow_Designability'].round(3)
    merged['QFlash_Designability'] = merged['QFlash_Designability'].round(3)
    
    match = re.search(r'epoch=?(\d+)', args.reqflash_path)
    epoch = match.group(1) if match else "unknown"
    out = f"results/designability_by_length_bins_epoch{epoch}_500steps.csv"
    if not os.path.exists('results'): os.makedirs('results')
    
    merged.to_csv(out, index=False)
        
    print(f"\nSaved comparison to {out}")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()
