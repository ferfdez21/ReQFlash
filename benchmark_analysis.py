import subprocess
import csv
import os
import argparse
import re

# Configuration
CONFIG = {
    "path": ".",
    "env_name": "reqflash",
    "output_csv": "reqflash-performance-scope.csv"
}



def run_benchmark(config, ckpt_path=None, num_timesteps=1, samples_per_sequence=1):
    """
    Runs the benchmark for a given configuration.
    """
    repo_path = config["path"]
    env_name = config["env_name"]
    output_csv = config["output_csv"]

    print(f"Generating {samples_per_sequence} proteins after {num_timesteps} inference steps...")
    
    stacked_records = []
    
    # Sequence lengths: 128, 256, ..., 524288
    # 524288 = 2^19. 128 = 2^7.
    # So we go from 2^7 to 2^19.
    
    sequence_length = 2**6
    max_length = 2**14
    
    while sequence_length <= max_length:
        print(f"  Testing sequence length: {sequence_length}")
        
        # Construct command
        # python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional
        cmd = [
            "conda", "run", "-n", env_name,
            "python", "-W", "ignore", "experiments/inference_se3_flows.py",
            "-cn", "inference_unconditional",
            f"inference.interpolant.sampling.num_timesteps={num_timesteps}",
            f"inference.samples.samples_per_length={samples_per_sequence}",
            f"inference.samples.min_length={sequence_length}",
            f"inference.samples.max_length={sequence_length}",
        ]
        
        if ckpt_path:
            cmd.append(f"inference.ckpt_path={ckpt_path}")
        
        # Set PYTHONPATH to include the repository root
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_path}:{env.get('PYTHONPATH', '')}"

        try:
            # Execute command
            # capture_output=True to suppress massive stdout/stderr unless needed for debugging
            # check=True to raise CalledProcessError on non-zero exit code
            # We use Popen to allow monitoring while it runs
            # Modified to capture stdout/stderr to find output directory
            with subprocess.Popen(cmd, cwd=repo_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                
                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    print(f"    Failed. Error type: subprocess.CalledProcessError (Exit code {proc.returncode})")
                    print(f"    Error output: {stderr.strip()}")
                    break
                
                # Parse output to find "Saving results to"
                output_dir = None
                full_output = (stdout or "") + "\n" + (stderr or "")
                match = re.search(r"Saving results to\s+(.+)", full_output)
                if match:
                    output_dir = match.group(1).strip()
                
                if output_dir:
                    csv_path = os.path.join(output_dir, "time_records.csv")
                    if os.path.exists(csv_path):
                        current_batch_records = []
                        with open(csv_path, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                # Extract columns: length, sample_time, memory_reserved
                                if all(k in row for k in ['length', 'sample_time', 'memory_reserved']):
                                    record = {
                                        'length': row['length'],
                                        'sample_time': row['sample_time'],
                                        'memory_reserved': row['memory_reserved']
                                    }
                                    stacked_records.append(record)
                                    current_batch_records.append(record)
                        
                        if current_batch_records:
                            times = [float(r['sample_time']) for r in current_batch_records]
                            mems = [float(r['memory_reserved']) for r in current_batch_records]
                            avg_time = sum(times) / len(times)
                            avg_mem = sum(mems) / len(mems)
                            print(f"    Avg Sample Time: {avg_time:.4f}s | Avg Memory Reserved: {avg_mem:.4f} GB")
                    else:
                        print(f"    Warning: time_records.csv not found in {output_dir}")
                else:
                    print("    Warning: Could not find output directory in logs.")

        except Exception as e:
            print(f"    An unexpected error occurred: {type(e).__name__}: {e}")
            break

        sequence_length *= 2



    # Save stacked time records
    stacked_csv_name = f"reqflash_performance_scope_{num_timesteps}step_{samples_per_sequence}samples.csv"
    print(f"Saving stacked records to {stacked_csv_name}")
    with open(stacked_csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["length", "sample_time", "memory_reserved"])
        writer.writeheader()
        writer.writerows(stacked_records)

def main():
    parser = argparse.ArgumentParser(description="Benchmark ReQFlash inference.")
    parser.add_argument("--ckpt_path", type=str, default="scope", help="Path to the checkpoint file.")
    parser.add_argument("--num_timesteps", type=int, default=1, help="Number of timesteps.")
    parser.add_argument("--samples_per_sequence", type=int, default=1, help="Number of samples per sequence length.")
    args = parser.parse_args()

    if args.ckpt_path == "scope":
        args.ckpt_path = "ckpts/reqflash/reqflash_train_scope_rectify/2025-12-12_12-25-07/last.ckpt"

    run_benchmark(CONFIG, ckpt_path=args.ckpt_path, num_timesteps=args.num_timesteps, samples_per_sequence=args.samples_per_sequence)
    print("\nBenchmarking complete.")

if __name__ == "__main__":
    main()
