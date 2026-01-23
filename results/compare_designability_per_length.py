import pandas as pd
import argparse
import os
import re
import numpy as np

def load_data(path, multi_ckpt):
    df_list = []
    
    if multi_ckpt:
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found")
            return pd.DataFrame()
            
        # Scan subdirectories
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"Multi-ckpt: Found {len(subdirs)} subdirectories in {path}")
        
        for ckpt in subdirs:
            csv_path = os.path.join(ckpt, "All_Results_Origin.csv")
            if os.path.exists(csv_path):
                try:
                    d = pd.read_csv(csv_path)
                    df_list.append(d)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
    else:
        # Single file/dir mode
        # User passes FOLDER, script appends "All_Results_Origin.csv"
        csv_path = os.path.join(path, "All_Results_Origin.csv")
        if os.path.exists(csv_path):
            try:
                d = pd.read_csv(csv_path)
                df_list.append(d)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Warning: {csv_path} not found")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def compute_metrics_binned(df, num_bins=8):
    if df.empty:
        return pd.DataFrame()
        
    if 'length' not in df.columns or 'min_rmsd' not in df.columns:
        print(f"Warning: columns missing in DataFrame")
        return pd.DataFrame()
        
    # Dynamic Binning
    min_len = 60
    max_len = 129
    
    dmin = df['length'].min()
    dmax = df['length'].max()
    if dmin < min_len: min_len = int(dmin)
    if dmax >= max_len: max_len = int(dmax + 1)
    
    edges = np.linspace(min_len, max_len, num_bins + 1).astype(int)
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
    parser = argparse.ArgumentParser(description="Compare Designability by Length Bins (Experimental vs Baseline).")
    parser.add_argument("--experimental", required=True, help="Path to experimental model inference output folder")
    parser.add_argument("--baseline", required=True, help="Path to baseline model inference output folder")
    parser.add_argument("--multi-ckpt", action="store_true", help="Aggregate multiple checkpoints in the given folders")
    args = parser.parse_args()
    
    num_bins = 8
    
    print(f"Loading experimental data from {args.experimental}...")
    df_raw_experimental = load_data(args.experimental, args.multi_ckpt)
    df_experimental = compute_metrics_binned(df_raw_experimental, num_bins=num_bins)
    
    print(f"Loading baseline data from {args.baseline}...")
    df_raw_baseline = load_data(args.baseline, args.multi_ckpt)
    df_baseline = compute_metrics_binned(df_raw_baseline, num_bins=num_bins)
    
    if df_experimental.empty or df_baseline.empty:
        print("Data extraction failed for one or both models.")
        return

    # Merge
    df_experimental = df_experimental.rename(columns={'designability': 'Experimental_Designability', 'scRMSD_str': 'Experimental_scRMSD'})
    df_baseline = df_baseline.rename(columns={'designability': 'Baseline_Designability', 'scRMSD_str': 'Baseline_scRMSD'})
    
    merged = pd.merge(df_baseline, df_experimental, on='length_bin', how='outer')
    
    cols = ['length_bin', 'Baseline_Designability', 'Baseline_scRMSD', 'Experimental_Designability', 'Experimental_scRMSD']
    # Filter only existing cols
    cols = [c for c in cols if c in merged.columns]
    merged = merged[cols]
    
    # Sort bins numerically by lower bound
    merged['sort_key'] = merged['length_bin'].astype(str).str.extract(r'^(\d+)').fillna(0).astype(int)
    merged = merged.sort_values(by='sort_key')
    merged = merged.drop(columns=['sort_key'])
    
    if 'Baseline_Designability' in merged.columns:
        merged['Baseline_Designability'] = merged['Baseline_Designability'].round(3)
    if 'Experimental_Designability' in merged.columns:
        merged['Experimental_Designability'] = merged['Experimental_Designability'].round(3)
    
    experimental_name = os.path.basename(os.path.normpath(args.experimental))
    baseline_name = os.path.basename(os.path.normpath(args.baseline))
    
    if args.multi_ckpt:
        out = f"results/compare_bins_multi_{experimental_name}_vs_{baseline_name}.csv"
    else:
        out = f"results/compare_bins_{experimental_name}_vs_{baseline_name}.csv"
        
    if not os.path.exists('results'): os.makedirs('results')
    
    merged.to_csv(out, index=False)
        
    print(f"\nSaved comparison to {out}")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()
