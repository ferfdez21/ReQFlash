import wandb
import pandas as pd
import os
import numpy as np

import argparse

# Averages for SCOPe
SCOPE_HELIX = 0.330
SCOPE_STRAND = 0.260

def main():
    parser = argparse.ArgumentParser(description="Select best checkpoint based on WandB metrics.")
    parser.add_argument("--run_id", type=str, default="ous7dfjg", help="WandB Run ID")
    parser.add_argument("--project", type=str, default="reqflash", help="WandB Project")
    parser.add_argument("--entity", type=str, default=None, help="WandB Entity")
    parser.add_argument("--ckpt_dir", type=str, default="/home/ffernandez/Desktop/code/ReQFlash/ckpts/qflash_scope", help="Checkpoint directory")
    args = parser.parse_args()
    
    api = wandb.Api()
    
    run_id = args.run_id # Defaulting to the one we found matches
    project = args.project
    entity = args.entity
    
    try:
        if entity:
            path = f"{entity}/{project}/{run_id}"
        else:
            # Try to resolve path by listing or just formatted string if allowed
            # api.run() takes "entity/project/run_id"
            # If entity is None, we need to find it.
             path = f"{project}/{run_id}" # This might assume default entity
             
        # Robust way: use api.run() which takes a path.
        # But if we don't know entity, we have to search.
        
        # Let's try searching if entity not provided
        if not entity:
             # Search in project
             runs = api.runs(project)
             target_run = next((r for r in runs if r.id == run_id), None)
             if not target_run: 
                 # Try with user's default entity
                 print(f"Run {run_id} not found in project {project} (default entity). Checking explicit... ")
                 raise ValueError("Run not found")
        else:
             target_run = api.run(path)

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        # Fallback logic from before
        try:
             runs = api.runs(path=project)
             target_run = next((r for r in runs if r.id == run_id), None)
        except:
             print("Could not find run.")
             return

    if not target_run:
        print(f"Run {run_id} not found.")
        return
            
    if not target_run:
        print(f"Run {run_id} not found in project.")
        return
        
    print(f"Found Run: {target_run.name} ({target_run.id})")
    
    # Get history
    # keys needed: valid/ca_ca_valid_percent, valid/helix_percent, valid/strand_percent, epoch
    history = target_run.history(keys=[
        'valid/ca_ca_valid_percent', 
        'valid/helix_percent', 
        'valid/strand_percent', 
        'epoch',
        'trainer/global_step'
    ], pandas=True)
    
    if history.empty:
        print("History is empty.")
        return
        
    print(f"Retrieved {len(history)} metrics records.")
    
    # Map step/epoch to Checkpoint
    # Checkpoint filenames: epoch=176-step=325149.ckpt
    # We have 'epoch' and 'trainer/global_step' in history.
    # Note: 'trainer/global_step' matches 'step=...' in filename.
    
    ckpt_dir = args.ckpt_dir
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint dir {ckpt_dir} does not exist.")
        return
        
    ckpt_files = os.listdir(ckpt_dir)
    
    # Build map of step -> filename
    step_to_file = {}
    for f in ckpt_files:
        if not f.endswith('.ckpt'): continue
        # Parse step
        parts = f.split('-')
        for p in parts:
            if 'step=' in p:
                try:
                    s = int(p.split('=')[1].replace('.ckpt',''))
                    step_to_file[s] = os.path.join(ckpt_dir, f)
                except:
                    pass
                    
    print(f"Found {len(step_to_file)} checkpoints locally.")
    
    # Merge history with available checkpoints
    # We filter history where 'trainer/global_step' is in step_to_file keys
    
    df = history.copy()
    df['step'] = df['trainer/global_step']
    
    # Match history step to checkpoint file with tolerance
    def find_ckpt(s):
        if s in step_to_file: return step_to_file[s]
        if s+1 in step_to_file: return step_to_file[s+1]
        if s-1 in step_to_file: return step_to_file[s-1]
        return None

    df['path'] = df['step'].apply(find_ckpt)
    
    # Drop rows where no checkpoint was found
    df.dropna(subset=['path'], inplace=True)
    
    if df.empty:
        print("No overlap between logged steps and local checkpoints (even with +/-1 tolerance).")
        # Print sample to debug
        print("History steps:", history['step'].tail().tolist())
        print("Ckpt steps:", list(step_to_file.keys())[:5])
        return

    # Filter Metrics
    # 1) ca_ca_valid_percent > 0.99
    # 2) sec_deviation < 0.2
    
    df['sec_deviation'] = np.sqrt(
        (df['valid/helix_percent'] - SCOPE_HELIX)**2 + 
        (df['valid/strand_percent'] - SCOPE_STRAND)**2
    )
    
    print("\nFiltering...")
    
    # 0) Epoch > 180
    filtered_0 = df[df['epoch'] > 180]
    print(f"0) Epoch > 180: {len(filtered_0)} candidates")
    
    filtered_1 = filtered_0[filtered_0['valid/ca_ca_valid_percent'] > 0.99]
    print(f"1) Ca-Ca > 0.99: {len(filtered_1)} candidates")
    
    filtered_2 = filtered_1[filtered_1['sec_deviation'] < 0.2]
    print(f"2) SecDev < 0.2: {len(filtered_2)} candidates")
    
    candidates = filtered_2
    if candidates.empty:
        print("No perfect candidates. Determining best available...")
        if not filtered_1.empty: candidates = filtered_1
        else: candidates = df
        
    # Sort by sec_deviation ascending (lower is better)
    sorted_candidates = candidates.sort_values(by='sec_deviation', ascending=True)
    
    # Take top 10
    top_10 = sorted_candidates.head(10)
    
    print("\n" + "="*80)
    print(f"TOP {len(top_10)} CHECKPOINTS SELECTED")
    print("="*80)
    print(f"{'Path':<80} | {'Step':<8} | {'Ca-Ca':<7} | {'Strand':<7} | {'SecDev':<7}")
    print("-" * 120)
    
    for idx, row in top_10.iterrows():
        print(f"{row['path']:<80} | {row['step']:<8.0f} | {row['valid/ca_ca_valid_percent']:<7.4f} | {row['valid/strand_percent']:<7.4f} | {row['sec_deviation']:<7.4f}")
        
    print("="*80)
    
    # Save CSV of top 10
    top_10.to_csv("results/top_10_checkpoints.csv", index=False)
    print("Saved results/top_10_checkpoints.csv")

if __name__ == "__main__":
    main()
