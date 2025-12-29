"""Script for preprocessing PDB files."""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md


from data import utils as du
from data import parsers
from data import errors


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=50)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str)
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true',
    default=False)
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_file(file_path: str, write_dir: str, type: str = 'sample'):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.
        type: sample or noise.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    dir_path = os.path.dirname(file_path)
    path_parts = dir_path.split(os.sep)

    third_level_folder = path_parts[-3]  # run_2025-01-12_18-00-31
    fourth_level_folder = path_parts[-2]  # length_61
    fifth_level_folder = path_parts[-1]  # sample_0

    pdb_name = f"{third_level_folder}_{fourth_level_folder}_{fifth_level_folder}"
    metadata['pdb_name'] = pdb_name
    metadata['type'] = type
    os.makedirs(os.path.join(write_dir, pdb_name), exist_ok=True)
    processed_path = os.path.join(write_dir, f'{pdb_name}/{type}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    
    try:
        # MDtraj
        traj = md.load(file_path)
        # SS calculation
        # SS denote Secondary Structure
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # RG calculation
        # Radius of gyration
        pdb_rg = md.compute_rg(traj)
    except Exception as e:
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_rg[0]
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            pdb_dir = os.path.dirname(file_path)
            metadata = process_file(
                os.path.join(pdb_dir, 'sample.pdb'),
                write_dir,
                type='sample')
            metadata_1 = process_file(
                os.path.join(pdb_dir, 'noise.pdb'),
                write_dir,
                type='noise')

            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
            all_metadata.append(metadata_1)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        pdb_dir = os.path.dirname(file_path)
        metadata = process_file(
            os.path.join(pdb_dir, 'sample.pdb'),
            write_dir,
            type='sample')
        metadata_1 = process_file(
            os.path.join(pdb_dir, 'noise.pdb'),
            write_dir,
            type='noise')
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return [metadata, metadata_1]
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def get_all_file_paths(base_dir):
    all_file_paths = []

    for root, dirs, files in os.walk(base_dir):
        if 'self_consistency' in dirs: 
            length_value = None
            for part in root.split(os.sep):
                if part.startswith('length_'):
                    length_value = int(part.split('_')[1])
                    break
            folder_name = os.path.basename(root)
            if length_value is not None:

                csv_path = os.path.join(root, 'self_consistency', 'sc_results.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)

                    max_tm_score = df['tm_score'].max()
                    min_rmsd = df['rmsd'].min()

                    if length_value > 400:
                        if max_tm_score > 0.9 or min_rmsd < 2:
                            sample_pdb_path = os.path.join(root, 'sample.pdb')
                            if os.path.exists(sample_pdb_path):
                                all_file_paths.append(sample_pdb_path)
                    else:
                        if min_rmsd < 2:
                            sample_pdb_path = os.path.join(root, 'sample.pdb')
                            if os.path.exists(sample_pdb_path):
                                all_file_paths.append(sample_pdb_path)

    return all_file_paths


def main(args):
    pdb_dir = args.pdb_dir
    if not os.path.isdir(pdb_dir):
        raise ValueError(f"{pdb_dir} is not a valid directory")
    
    # all_file_paths = get_all_file_paths(pdb_dir) #* For with self-consistency
    all_file_paths = [  #* For without self-consistency
                    os.path.join(root, 'sample.pdb')
                    for root, dirs, files in os.walk(pdb_dir)
                    if 'sample.pdb' in files and 'noise.pdb' in files
                ]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [item for sublist in all_metadata if sublist is not None for item in sublist]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')
    print(f'Metadata written to {metadata_path}')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)