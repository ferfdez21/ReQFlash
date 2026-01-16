Running foldseek with:
  Input PDB: /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/length_127/sample_0/sample.pdb
  Database: pdb
  Output alignment file: /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/length_127_sample_0_sample_aln.txt
  Temporary folder: /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/tmp_length_127_sample_0_sample
easy-search \
  /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/length_127/sample_0/sample.pdb \
  pdb \
  /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/length_127_sample_0_sample_aln.txt \
  /home/ffernandez/Desktop/code/ReQFlash/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis/10_steps/tmp_length_127_sample_0_sample \
  --alignment-type 1 \
  --exhaustive-search \
  --max-seqs 10000000000 \
  --tmscore-threshold 0.0 \
  --format-output query,target,alntmscore,lddt,evalue