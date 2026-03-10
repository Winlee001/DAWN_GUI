# DAWN Train GUI User Guide

This GUI is used to launch and monitor DAWN training jobs with a visual workflow (parameter editing, YAML import/export, live logs, loss/LR curves, and system monitor).

### TL;DR: Select a training script → load config → set `Run_description` → click **Start Training**.

---

## Compatibility & Installation

You can use either packaged executable or Python script mode.

1. **Run packaged EXE (recommended for non-developers)**
   - Launch `DAWN_TrainGUI.exe`.
   - If crashed unexpectedly, check `DAWN_TrainGUI_crash.log` in the same folder.

2. **Run Python script (recommended for development/debug)**
   - Use Python 3.10+ (project currently tested with 3.11).
   - Install dependencies from workspace root `requirements.txt`.
   - Required core packages for Train GUI:
     - `PyQt6`
     - `omegaconf` or `PyYAML`
     - `matplotlib` (for loss/LR plots)
     - `psutil` (+ `pynvml` for GPU monitor)
   - Start with:
     - `python DAWN_TrainGUI_code/Train_GUI.py`

3. **CUDA / GPU notes**
   - Training speed strongly depends on GPU.
   - GPU usage display requires NVIDIA driver + `pynvml`.

---

## Quick Start Guide

### Step 1: Select training script

Use **Training Script** dropdown:
- `gray_unpair_pretrain_mu.py` (single-channel workflow)
- `gray_dualchan_pretrain_mu.py` (dual-channel workflow)

When switching scripts, the GUI auto-loads matching template from `config_templates/`.
for single channel model,choose gray_unpair_pretrain_mu.py as training scipt which will load 'config_templates/DAWN_SingleChannel_train_template.yaml'
for dual channel model,choose gray_dualchan_pretrain_mu.py as training scipt which will load 'config_templates/DAWN_DualChannel_train_template.yaml'

### Step 2: Load config (optional but recommended)

You can configure parameters in 3 ways:
- **Manual edit**: directly edit form fields
- **Load YAML**: import `.yaml/.yml`
- **Import .sh**: parse command-line style shell script

You can save current form with **Save YAML**.

### Step 3: Run and monitor training

Click **Start Training**.

GUI behavior:
- Builds command automatically and runs selected script under `DAWN_gray/`
- Streams stdout/stderr into **Logs** tab
- Parses log file and updates:
  - Epoch / Step / Loss / LR status labels
  - **Loss** plot (Avg Loss)
  - **LR** plot
- System monitor panel updates CPU/RAM usage
- If GPU available, each GPU line shows util + memory usage
- If GPU monitor dependency is missing, monitor falls back gracefully
- Click **Stop** to request process termination.

---

## Output & Logging Rules

1. **Working directory for training process**
   - `DAWN_gray/`

2. **Log file path pattern**
   - `<log_dir>/<log_name>_Noisy_data_<Run_description>_log.txt`

3. **Relative `log_dir` handling**
   - If `log_dir` is relative (e.g. `ckpt`), it is resolved under `DAWN_gray/`

4. **Command visibility**
   - Full generated command is printed in the log panel at startup

---

## Field Notes (Most Used)

- Core fields to set first:
   - `trainset`, `valset`
   - `log_dir`
   - `bsn_ver`
   - `loss_choice`
   - `epoch`, `batch_size`, `lr_dbsn`
   - `Run_description` (**required**)
- Resume logic:
   - `resume = new`: start fresh training
   - `resume = continue`: must provide `last_ckpt` (`.pth`)
- Start checks:
   - Empty `Run_description` blocks start
   - `resume=continue` without `last_ckpt` blocks start

- `trainset`, `valset`: training/validation dataset names (comma-separated if in dual-channel mode)
- `input_channel`, `output_channel`, `middle_channel`: model IO/internal channels,you must keep input_channel equal to frame of train data
- `patch_size`, `batch_size`, `load_thread`: data loader and patch settings
- `steps`: LR schedule milestones (e.g. `30,60,80`)
- `bsn_ver`: model variant selector
- `loss_choice`: training loss mode
- `blindspot_conv_type`, `mask_shape`, `blindspot_conv_bias`: blind-spot settings
- `device_ids`: target GPU IDs (`all` or custom list depending on backend parser)
- `dynamic_load`, `no_flip`, `frame_shuffle`: boolean runtime flags

we recommend to main most settings in most case except trainset/valset/log_dir/Run_description.
---

## Troubleshooting

1. **No plots shown**
   - Install `matplotlib`; otherwise GUI shows fallback text in plot tabs.

2. **Cannot load YAML**
   - Install `omegaconf` or `PyYAML`.

3. **GPU line shows “Not available”**
   - Check NVIDIA driver and `pynvml` installation.

4. **Start button does nothing / immediate error**
   - Check required fields (`Run_description`, resume checkpoint path)

---

## Suggested Workflow

1. Choose script by task (single vs dual channel)
2. Load template YAML first
3. Only change minimal experiment-specific fields:
   - log_dir
   - dataset names
   - `Run_description`
   - core hyperparameters
4. Save YAML per experiment for reproducibility
5. Start from `resume=new`; use `resume=continue` only for checkpoint continuation

---

## Version

### Ver. 1.0 (Train GUI)
- Initial release:
  - Visual parameter editor
  - YAML load/save
  - `.sh` argument import
  - Live training logs
  - Loss/LR visualization
  - CPU/GPU system monitor
