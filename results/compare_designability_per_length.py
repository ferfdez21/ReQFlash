import pandas as pd
import argparse
import os
import re

def compute_metrics_binned(csv_path):
    # Reads All_Results_Origin.csv and bins by length
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if 'length' not in df.columns or 'min_rmsd' not in df.columns:
        print(f"Warning: columns missing in {csv_path}")
        return pd.DataFrame()
        
    # Binning
    # Range 60-128. 4 bins.
    bins = [60, 77, 94, 111, 129] # custom edges approx quarters: 60-77, 77-94...
    labels = ["60-77", "78-94", "95-111", "112-128"]
    
    # Actually let's use pd.cut with equal width or explicit edges?
    # User said "separate the whole length range (i.e., 60 to 128) in 4 bins or quarters"
    # pd.cut(df['length'], bins=4) is the easiest way to get equal width intervals.
    # explicit bins might be cleaner for display.
    # Let's use clean edges. 60 + 17 = 77. 77+17=94. 94+17=111. 111+17=128.
    bins = [59, 77, 94, 111, 129] # Edges. (59, 77] includes 60..77.
    labels = ["60-77", "78-94", "95-111", "112-128"]
    
    df['bin'] = pd.cut(df['length'], bins=bins, labels=labels)
    
    results = []
    grouped = df.groupby('bin', observed=False) # observed=False to include empty bins if any? No, default is fine.
    
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
    
    flash_csv = os.path.join(args.reqflash_path, "500_steps", "All_Results_Origin.csv")
    flow_csv = "inference_outputs/ckpts/qflow_scope/500_steps/All_Results_Origin.csv"
    
    print(f"Reading QFlash: {flash_csv}...")
    df_flash = compute_metrics_binned(flash_csv)
    
    print(f"Reading QFlow: {flow_csv}...")
    df_flow = compute_metrics_binned(flow_csv)
    
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
    out = f"results/designability_by_length_bins_epoch{epoch}_500steps.txt"
    if not os.path.exists('results'): os.makedirs('results')
    
    with open(out, 'w') as f:
        f.write(merged.to_string(index=False))
        
    print(f"\nSaved comparison to {out}")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()
