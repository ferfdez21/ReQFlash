import pandas as pd
import argparse
import os
import re

def compute_metrics_per_length(csv_path):
    # Reads All_Results_Origin.csv
    # Calculate designability (min_rmsd < 2) per length
    # Calculate min_rmsd mean/std per length
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Check necessary columns
    if 'length' not in df.columns or 'min_rmsd' not in df.columns:
        print(f"Warning: columns missing in {csv_path}")
        return pd.DataFrame()
        
    results = []
    grouped = df.groupby('length')
    for length, group in grouped:
        designability = (group['min_rmsd'] < 2).mean()
        mean_rmsd = group['min_rmsd'].mean()
        std_rmsd = group['min_rmsd'].std()
        
        results.append({
            'length': length,
            'designability': designability,
            'scRMSD_str': f"{mean_rmsd:.3f} ± {std_rmsd:.3f}"
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Compare Designability by Length (500 steps).")
    parser.add_argument("reqflash_path", help="Path to ReQFlash epoch folder (e.g. .../epoch102)")
    args = parser.parse_args()
    
    # Paths hardcoded relative to args or workspace
    # 500 steps fixed
    flash_csv = os.path.join(args.reqflash_path, "500_steps", "All_Results_Origin.csv")
    
    # Baseline path (assuming parallelism with experimental setup or as found previously)
    # Found previously at: inference_outputs/ckpts/qflow_scope/500_steps/All_Results_Origin.csv
    flow_csv = "inference_outputs/ckpts/qflow_scope/500_steps/All_Results_Origin.csv"
    
    print(f"Reading QFlash: {flash_csv}...")
    df_flash = compute_metrics_per_length(flash_csv)
    
    print(f"Reading QFlow: {flow_csv}...")
    df_flow = compute_metrics_per_length(flow_csv)
    
    if df_flash.empty or df_flow.empty:
        print("Data extraction failed.")
        return

    # Merge
    # Rename columns
    df_flash = df_flash.rename(columns={'designability': 'QFlash_Designability', 'scRMSD_str': 'QFlash_scRMSD'})
    df_flow = df_flow.rename(columns={'designability': 'QFlow_Designability', 'scRMSD_str': 'QFlow_scRMSD'})
    
    merged = pd.merge(df_flow, df_flash, on='length', how='outer')
    merged = merged.sort_values(by='length')
    
    # Format
    cols = ['length', 'QFlow_Designability', 'QFlow_scRMSD', 'QFlash_Designability', 'QFlash_scRMSD']
    merged = merged[cols]
    
    # Round designability
    merged['QFlow_Designability'] = merged['QFlow_Designability'].round(3)
    merged['QFlash_Designability'] = merged['QFlash_Designability'].round(3)
    
    # Save
    match = re.search(r'epoch=?(\d+)', args.reqflash_path)
    epoch = match.group(1) if match else "unknown"
    out = f"results/designability_by_length_epoch{epoch}_500steps.txt"
    if not os.path.exists('results'): os.makedirs('results')
    
    with open(out, 'w') as f:
        f.write(merged.to_string(index=False))
        
    print(f"\nSaved comparison to {out}")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()
