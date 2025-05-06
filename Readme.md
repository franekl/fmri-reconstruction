# fmri-reconstruction (MindEyeV2 variant)

## Team

- **Albert Steenstrup**  
- **Franek Liszka**   

**Institution**: [IT University of Copenhagen](https://en.itu.dk)
**Contact**: `{frli, albst}@itu.dk`

---

## Project Overview
We reconstruct natural images from fMRI recordings by mapping voxel activations → CLIP embedding space → image latents → decoded images.

Rather than the full-scale MindEyeV2, we adapt for weaker compute:
- Reduced hidden dimension (hidden_dim=1024 vs. original 4096)  
- Smaller batch sizes (e.g. 4–16 instead of 24+)  
- Switched CLIP backbone from ViT-L-14 to ViT-B-32  
- Adjusted CLIP token dims: clip_seq_dim=49, clip_emb_dim=768 (down from 256×1664)  
- Ran text-model inference on CPU to avoid OOM, using minibatches of 4  
- Trained on a single session (~750 trials, ~1 hour) for one subject  

Hardware: initially on an NVIDIA L4 (24 GB) via Lightning.ai Studio, then on Google Cloud Services  
Logging: all runs tracked via Weights & Biases

This project is a lightweight adaptation of [MindEyeV2](https://github.com/MedARC-AI/MindEyeV2/tree/main), originally developed by MedARC.

---

## Goals

### Low-Compute Adaptation
Adapt the full MindEyeV2 pipeline to run on a single 24 GB GPU (NVIDIA L4) with reduced hidden dimensions, batch sizes, and a smaller CLIP backbone, while completing a full train + inference cycle in under a few hours.

### Single-Session Reconstruction
Train a subject-specific model on only 1 hour of fMRI data (~750 trials) and demonstrate that high-quality image reconstructions and caption predictions remain achievable.

### Architecture & Hyperparameter Exploration
Systematically vary and justify non-standard choices—hidden_dim (4096 → 1024), clip_seq_dim/clip_emb_dim (256×1664 → 49×768), disabling blurry_recon, CPU offload for text model—to identify configurations that balance performance and resource usage.

---

## Repository Structure & Code Summary

All code lives under src/ and is organized into four main Jupyter notebooks:

1. **Train.ipynb**  
   • Loads NSD webdataset & HDF5 betas  
   • Builds an MLP-Mixer backbone + optional diffusion prior  
   • Uses accelerate with DeepSpeed Stage 2 CPU offload  
   • Example arguments for low-compute run:  
     ```
     --hidden_dim=1024 --batch_size=4 --num_sessions=1 \
     --clip_seq_dim=49 --clip_emb_dim=768 --no-blurry_recon \
     --subj=1 --no-multi_subject
     ```  
   • Includes CUDA availability assertions and gradient checks  

2. **recon_inference.ipynb**  
   • Loads trained checkpoint on CPU  
   • Prepares test-set voxels & image indices  
   • Runs backbone → CLIP embeddings → diffusion prior → unCLIP decode  
   • Saves tensors:  
     - `*_all_recons.pt` (raw reconstructions)  
     - `*_all_blurryrecons.pt` (VAE reconstructions)  
     - `*_all_predcaptions.pt` (GIT captions)  
     - `*_all_clipvoxels.pt`  

3. **enhanced_recon_inference.ipynb**  
   • Applies SD-XL refinement to `all_recons` for higher-resolution outputs  
   • Saves `*_all_enhancedrecons.pt`  

4. **final_evaluations.ipynb**  
   • Loads reconstructions, ground truth images & captions  
   • Creates visual grids (original vs. reconstruction)  
   • Computes metrics (PixCorr, SSIM, retrieval accuracy, caption scores, brain-correlation)  
   • Exports CSVs under `tables/`  

---

## Installation

Agree to the [Natural Scenes Dataset’s Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform?pli=1).


Clone this repository and enter the `src/` directory:
```bash
git clone https://github.com/franekl/fmri-reconstruction.git
cd fmri-reconstruction/src
```

The full pscotti/mindeyev2 dataset is over 100 GB. To fetch only the minimal files needed for training/inference/evaluation, run:
```python
import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "pscotti/mindeyev2"
branch = "main"
exclude_dirs = ["train_logs", "evals"]
exclude_files = ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]
include_specific_files = [
    "evals/all_images.pt",
    "evals/all_captions.pt",
    "evals/all_git_generated_captions.pt"
]

def download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files):
    files = list_repo_files(repo_id, repo_type="dataset", revision=branch)
    for file_path in files:
        skip_dir = any(d in file_path for d in exclude_dirs)
        skip_file = any(f in file_path for f in exclude_files)
        if (not skip_dir or file_path in include_specific_files) and not skip_file:
            hf_hub_download(
                repo_id,
                filename=file_path,
                repo_type="dataset",
                revision=branch,
                local_dir=os.path.join(os.getcwd(), "data")
            )

download_files(repo_id, branch, exclude_dirs, exclude_files, include_specific_files)
```

---

## Environment Setup

```bash
cd src
./setup.sh    # creates fmri conda environment
source fmri/bin/activate
```

---

## Running the Pipeline

Example: 1-session train → recon → refine → eval

```bash
jupyter nbconvert --to notebook --execute Train.ipynb -- \
    --hidden_dim=1024 --batch_size=4 --num_sessions=1 \
    --clip_seq_dim=49 --clip_emb_dim=768 --subj=1 --no-multi_subject

jupyter nbconvert --to notebook --execute recon_inference.ipynb
jupyter nbconvert --to notebook --execute enhanced_recon_inference.ipynb
jupyter nbconvert --to notebook --execute final_evaluations.ipynb
```

---

## Outputs

- Reconstructions: `evals/{model_name}/`  
- Metrics CSVs: `tables/`  
- Comparison images: saved `.png` in repo root  

---

## References

Scotti, P., Tripathy, S., Torrico, O., Kneeland, P., Chen, Z., Narang, S., … Abraham, A. (2024). MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. *International Conference on Machine Learning*. arXiv:2403.11207

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, S., Dowdle, L. T., … Kay, K. N. (2021). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*  

Original repository: [MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2/tree/main)
