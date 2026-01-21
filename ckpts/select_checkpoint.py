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
        'valid/num_ca_ca_clashes',
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

    # --- Convergence Detection ---
    print("Fetching training loss history for convergence detection...")
    # Fetch sampled history for loss to be efficient (e.g. 2000 points)
    try:
        loss_df = target_run.history(
            keys=['train/loss', 'trainer/global_step'], 
            samples=2000,
            pandas=True
        )
    except Exception as e:
        print(f"Warning: Could not fetch loss history: {e}. defaulting to epoch > 180 strategy.")
        loss_df = pd.DataFrame()

    convergence_step = 0
    if not loss_df.empty and 'train/loss' in loss_df.columns:
        loss_df = loss_df.dropna(subset=['train/loss']).sort_values('trainer/global_step')
        
        # Smooth the loss curve to remove noise
        # Since we have ~2000 sampled points, a window of ~50 is reasonable (~2.5% of data)
        loss_df['smoothed_loss'] = loss_df['train/loss'].rolling(window=50, center=True).mean()
        
        # Drop NaNs created by rolling
        clean_loss = loss_df.dropna(subset=['smoothed_loss'])
        
        if len(clean_loss) > 10:
            # Knee/Elbow Detection (Kneedle algorithm concept)
            # Normalize to [0, 1]
            steps = clean_loss['trainer/global_step'].values
            losses = clean_loss['smoothed_loss'].values
            
            # We focus on the curve. Normalization is crucial.
            step_min, step_max = steps.min(), steps.max()
            loss_min, loss_max = losses.min(), losses.max()
            
            x_norm = (steps - step_min) / (step_max - step_min)
            # Loss decreases, so we invert it to make it an increasing "knee" or work with the "L" shape
            # Standard Kneedle finds the point furthest from the line connecting start and end.
            # Line from (x0, y0) to (xN, yN)
            y_norm = (losses - loss_min) / (loss_max - loss_min)
            
            # Vector from start to end
            # P0 = (0, y_norm[0])
            # P1 = (1, y_norm[-1]) 
            # Note: typically loss goes High -> Low. 
            # y_norm[0] is roughly 1.0 (highest loss), y_norm[-1] is 0.0 (lowest loss).
            
            # Distance from point (x,y) to line P0-P1:
            # Line eq: (y1-y0)x - (x1-x0)y + x1y0 - y1x0 = 0
            # Here: (y_end - y_start)*x - (1-0)*y + ...
            
            y0, y1 = y_norm[0], y_norm[-1]
            # Normal vector to line (dy, -dx) -> (y1-y0, -1)
            # But simpler: calculate vertical distance to the secant line if x is monotonic
            # Secant line y_sec(x) = y0 + (y1 - y0) * x
            # Since loss is convex-ish (L-shape), the point maximizing |y_norm - y_sec| is the knee.
            
            y_secant = y0 + (y1 - y0) * x_norm
            distances = np.abs(y_norm - y_secant)
            
            knee_idx = np.argmax(distances)
            convergence_step = steps[knee_idx]
            
            print(f"Detected convergence 'knee' at step {convergence_step} "
                  f"(Loss: {losses[knee_idx]:.4f})")
    
    # -----------------------------

    # Filter Metrics
    # 0) valid/num_ca_ca_clashes == 0
    # 1) ca_ca_valid_percent > 0.99
    # 2) sec_deviation < 0.2
    
    df['sec_deviation'] = np.sqrt(
        (df['valid/helix_percent'] - SCOPE_HELIX)**2 + 
        (df['valid/strand_percent'] - SCOPE_STRAND)**2
    )
    
    print("\nFiltering...")

    # 0) Clashes == 0
    if 'valid/num_ca_ca_clashes' in df.columns:
        filtered_0 = df[df['valid/num_ca_ca_clashes'] == 0]
        print(f"0) Clashes == 0: {len(filtered_0)} candidates")
    else:
        print("Warning: 'valid/num_ca_ca_clashes' not found in metrics. Skipping clash filter.")
        filtered_0 = df
    
    # 1) Convergence Filter
    if convergence_step > 0:
        filtered_1 = filtered_0[filtered_0['step'] >= convergence_step]
        print(f"1) Convergence (Step >= {convergence_step}): {len(filtered_1)} candidates")
    else:
        # Fallback to previous heuristic
        print("1) No convergence detected, falling back to Epoch > 180")
        filtered_1 = filtered_0[filtered_0['epoch'] > 180]
        print(f"1) Epoch > 180: {len(filtered_1)} candidates")
    
    filtered_2 = filtered_1[filtered_1['valid/ca_ca_valid_percent'] > 0.99]
    print(f"2) Ca-Ca > 0.99: {len(filtered_2)} candidates")
    
    filtered_3 = filtered_2[filtered_2['sec_deviation'] < 0.2]
    print(f"3) SecDev < 0.2: {len(filtered_3)} candidates")
    
    candidates = filtered_3
    if candidates.empty:
        print("No perfect candidates. Determining best available...")
        if not filtered_2.empty: candidates = filtered_2
        elif not filtered_1.empty: candidates = filtered_1
        else: candidates = filtered_0 # Fallback to clash-free
        
    # Sort by sec_deviation ascending (lower is better)
    sorted_candidates = candidates.sort_values(by='sec_deviation', ascending=True)
    
    # Take top 10
    top_10 = sorted_candidates.head(10)
    
    print("\n" + "="*90)
    print(f"TOP {len(top_10)} CHECKPOINTS SELECTED")
    print("="*90)
    print(f"{'Path':<80} | {'Step':<8} | {'Ca-Ca':<7} | {'Strand':<7} | {'Clashes':<7} | {'SecDev':<7}")
    print("-" * 130)
    
    for idx, row in top_10.iterrows():
        clashes = row.get('valid/num_ca_ca_clashes', -1)
        print(f"{row['path']:<80} | {row['step']:<8.0f} | {row['valid/ca_ca_valid_percent']:<7.4f} | {row['valid/strand_percent']:<7.4f} | {clashes:<7.1f} | {row['sec_deviation']:<7.4f}")
        
    print("="*80)
    
    # Save CSV of top 10
    top_10.to_csv("results/top_10_checkpoints_test.csv", index=False)
    print("Saved results/top_10_checkpoints_test.csv")

if __name__ == "__main__":
    main()
