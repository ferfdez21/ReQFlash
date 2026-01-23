import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

def parse_metric_score(ckpt_path):
    metrics_path = os.path.join(ckpt_path, "Metrics.txt")
    if not os.path.exists(metrics_path):
        return -1.0
    
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    # grouped_rmsd_below_2_ratio: 0.787 ± 0.152
    m = re.search(r'grouped_rmsd_below_2_ratio:\s*([0-9.]+)', content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return -1.0
    return -1.0

def get_ranked_checkpoints(root_dir):
    checkpoints = []
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return []

    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for path in subdirs:
        score = parse_metric_score(path)
        if score >= 0:
            checkpoints.append({
                'name': os.path.basename(path),
                'path': path,
                'score': score
            })
            
    # Sort descending by score
    checkpoints.sort(key=lambda x: x['score'], reverse=True)
    return checkpoints

def get_designability_curve(ckpt_path):
    csv_path = os.path.join(ckpt_path, "All_Results_Origin.csv")
    if not os.path.exists(csv_path):
        print(f"CSV missing for {ckpt_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # Ensure we have numeric length and min_rmsd
        if 'length' not in df.columns or 'min_rmsd' not in df.columns:
            return None
        
        # Calculate designability: 1 if min_rmsd < 2, else 0
        df['designable'] = df['min_rmsd'] < 2.0
        
        # Group by length and mean
        curve = df.groupby('length')['designable'].mean()
        return curve
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare Top 3 Checkpoints per Length Matrix")
    parser.add_argument("experimental_dir", help="Experimental model checkpoints directory")
    parser.add_argument("baseline_dir", help="Baseline model checkpoints directory")
    args = parser.parse_args()
    
    # 1. Get Top 3
    top_experimental = get_ranked_checkpoints(args.experimental_dir)[:3]
    top_baseline = get_ranked_checkpoints(args.baseline_dir)[:3]
    
    if len(top_experimental) < 3 or len(top_baseline) < 3:
        print(f"Not enough checkpoints found. Experimental: {len(top_experimental)}, Baseline: {len(top_baseline)}")
        # Proceed with what we have? The user asked for 3x3.
        # We will cycle or handle gracefully, but strictly we need 3.
        # Let's pad with None if necessary to keep 3x3 grid logic simple, but warn.
        pass

    print("Top 3 Experimental:")
    for c in top_experimental: print(f"  {c['name']} ({c['score']})")
    print("Top 3 Baseline:")
    for c in top_baseline: print(f"  {c['name']} ({c['score']})")
    
    # 2. Prepare Data
    # Pre-load curves to avoid reading repeatedly
    experimental_curves = [get_designability_curve(c['path']) for c in top_experimental]
    baseline_curves = [get_designability_curve(c['path']) for c in top_baseline]
    
    # 3. Plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
    fig.suptitle("Designability vs Length: Top 3 Comparison Matrix", fontsize=20)
    
    # Rows: GARFlow Ranks (0, 1, 2)
    # Cols: GAFL Ranks (0, 1, 2)
    
    for i in range(3): # Row (Experimental Rank i+1)
        for j in range(3): # Col (Baseline Rank j+1)
            ax = axes[i, j]
            
            # Plot Experimental
            if i < len(top_experimental) and experimental_curves[i] is not None:
                lbl = f"Experimental #{i+1}\n{top_experimental[i]['name'][:20]}..."
                ax.plot(experimental_curves[i].index, experimental_curves[i].values, label=lbl, color='blue', marker='o', markersize=3, alpha=0.7)
                exp_score = top_experimental[i]['score']
            else:
                exp_score = 0.0
            
            # Plot Baseline
            if j < len(top_baseline) and baseline_curves[j] is not None:
                lbl = f"Baseline #{j+1}\n{top_baseline[j]['name'][:20]}..."
                ax.plot(baseline_curves[j].index, baseline_curves[j].values, label=lbl, color='orange', marker='s', markersize=3, alpha=0.7)
                base_score = top_baseline[j]['score']
            else:
                base_score = 0.0
                
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.set_title(f"E#{i+1} ({exp_score:.3f})  vs  B#{j+1} ({base_score:.3f})", fontsize=10)
            
            if i == 2:
                ax.set_xlabel("Length")
            if j == 0:
                ax.set_ylabel("Designability Ratio")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "designability_matrix_top3.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
