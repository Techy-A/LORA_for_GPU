# LoRA for GPU: Voxtral QLoRA fine-tuning (spoken grammar scoring)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch%20CUDA%2012.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PEFT](https://img.shields.io/badge/QLoRA%20%2F%20PEFT-2c9fbf?style=flat-square)
![GPU](https://img.shields.io/badge/8%20GB%20GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)

Fine-tunes Voxtral-Mini-3B (an audio language model) to predict a 0-5 grammar score from a spoken English answer.
The model is loaded in 4-bit (QLoRA), so it fits an 8 GB GPU such as an RTX 5060. Only small LoRA adapters and a
linear score head are trained. The output is one file of predictions that gets blended with other models in
[shl-grammar-scoring](https://github.com/Techy-A/shl-grammar-scoring), which placed 2nd on the competition's public leaderboard.

## What you need

| | Minimum |
|---|---|
| GPU | NVIDIA, 8 GB VRAM (tested target: RTX 5060) |
| RAM | 16 GB (all audio is held in memory, about 4 GB) |
| Disk | 25 GB free (PyTorch about 3 GB, model about 9 GB, data about 2 GB) |
| Software | Windows 10/11, Python 3.11 or 3.12, a recent NVIDIA driver (570 or newer for RTX 50xx) |
| Time | About 45 min setup, then about 45 min per fold, 5 folds (3 to 5 hours in total) |

## Step by step

### 1. Get the code and the data
```powershell
git clone https://github.com/Techy-A/LORA_for_GPU.git
cd LORA_for_GPU
```
The audio data is **not** in this repo (competition data must not be redistributed). Aryan shares the
`Dataset_Final` folder separately (USB or Drive). Put it here so this file exists:
```
LORA_for_GPU\data\Dataset_Final\train.csv
LORA_for_GPU\data\Dataset_Final\test.csv
LORA_for_GPU\data\Dataset_Final\train\*.wav
LORA_for_GPU\data\Dataset_Final\test\*.wav
```

### 2. Install (once, about 15-30 min)
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 setup
```
Creates `.venv`, installs PyTorch for CUDA 12.8 (required for RTX 50xx cards) and `requirements.txt`.

### 3. Check the setup (1 min)
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 check
```
Must print the GPU name, its memory and `setup OK`.

### 4. Smoke test (about 10 min, the first run also downloads the 9 GB model)
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 smoke
```
Trains on 16 clips for 1 epoch. It must end with `fold 0 done, peak VRAM ... GB`. The RMSE here means nothing.

### 5. Fold 0 (about 45 min): the go / no-go test
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 fold0
```
Watch the line `fold 0 epoch 2: val RMSE ...` at the end and send it to Aryan:
* below about 0.56: promising, continue with step 6
* above about 0.62: stop, it is not helping

### 6. Folds 1-4 (about 3 hours) and merge
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 rest
powershell -ExecutionPolicy Bypass -File run.ps1 merge
```
`merge` prints `s5v_voxtral_qlora: CV RMSE ... | Pearson ...` and writes
`artifacts\pred_s5v_voxtral_qlora.parquet`.

### 7. Send the result back
```powershell
git add artifacts/pred_s5v_voxtral_qlora.parquet logs
git commit -m "QLoRA predictions, 5 folds"
git push
```
(or send that one `.parquet` file and the `logs` folder by Drive).

Each stage can be stopped and rerun on its own; the folds are saved separately (`artifacts\s5v_fold<k>.parquet`).
Keep the laptop plugged in, on the "Best performance" power mode, and stop it from sleeping.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA error: no kernel image is available` | PyTorch is too old for the GPU. Rerun step 2 (it installs the CUDA 12.8 build). |
| `CUDA out of memory` | In `src/15_voxtral_qlora.py` set `CROP = 20 * SR`, close other GPU apps, rerun the stage. |
| `running scripts is disabled on this system` | Use the full `powershell -ExecutionPolicy Bypass -File run.ps1 ...` form above. |
| `data/Dataset_Final missing` | Check the folder layout in step 1. |
| bitsandbytes errors on Windows | `.venv\Scripts\pip install -U bitsandbytes`, then rerun `check`. |
| Model download stalls | Rerun the stage; the download resumes. |

## What the script does

`src/15_voxtral_qlora.py`
* Loads Voxtral-Mini-3B in 4-bit NF4 and adds LoRA adapters (r = 16) to the attention projections of its language
  model; the audio encoder stays frozen.
* Each clip is followed by the grammar rubric. The score head reads the middle layer: the mean over the audio tokens
  plus the last position, then a linear layer.
* Trains on random 30 s crops for 3 epochs (learning rate 1e-4, head 1e-3, batch 2), speaker-grouped 5-fold
  cross-validation (`artifacts/folds.csv`), so the out-of-fold predictions are honest.
* Validation clips are scored on fixed test-length crops; test clips are scored whole, in 30 s windows.

`src/common.py` holds the shared paths, folds and training loop.

## Files

| File | Purpose |
|---|---|
| `run.ps1` | One command per stage (setup, check, smoke, fold0, rest, merge) |
| `requirements.txt` | Python packages (PyTorch is installed separately by `run.ps1 setup`) |
| `src/15_voxtral_qlora.py` | The fine-tuning script |
| `src/common.py` | Shared settings and training loop |
| `artifacts/folds.csv` | The fixed speaker-grouped folds (do not change) |

---

<sub>Built by [Aryan Awasthi](https://github.com/Techy-A). ENC @ Thapar Institute of Engineering and Technology.</sub>
