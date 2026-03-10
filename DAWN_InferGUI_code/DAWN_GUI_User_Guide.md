# DAWN GUI Manual

This is a GUI application for our paper **"DAWN: A Spatiotemporal Self-supervised Framework for Reconstructing Ultra–Low-Photon Single-Molecule Fluorescence Images"**.This method is compatible with both EMCCD and sCMOS detectors and extends seamlessly to live-cell imaging, where it reliably reconstructs molecular diffusion behavior even at extremely low photon flux. We aim to use this application to easily enhance the quality of videos captured by EMCCD or SCMOS cameras.


### TL;DR: Just click the "Reconstruct" button to enhance the preloaded videos. 


## Compatibility & Installation

If you have a GPU which supports CUDA >= 12.1, we recommend **running binary application on GPU**. Otherwise please **run binary application on CPU**. If you have a GPU with CUDA < 12.1 or you want custom environment settings, please **run the python script** instead (you may still be able to run the binary but is not guaranteed).

1. **Running binary application on CPU**
   - No additional requirements needed.
   - Double-click the "MUFFLE_GUI.exe" file to launch the GUI.
   - **Note: The reconstruction process will be extremely slow on CPU. Not recommended for formal use.**
2. **Running binary application on GPU with CUDA >= 12.1 support (recommended)**
   - Make sure that the GPU device has CUDA support. Check [here](https://developer.nvidia.com/cuda-gpus) for a list of supported GPUs.
   - Install CUDA >= 12.1 from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64).
   - Double-click the "DAWN_GUI.exe" file to launch the GUI.
3. **Running the python script**
   - refer to DAWN_InferGUI_code for GUI code and DAWN_gray for source code

---


## Quick Start Guide

Short workflow from input to output.

### Step 1: Select Processing Mode

- **Single Channel**: one TIFF video in, one enhanced TIFF out.
- **Dual Channel**: two TIFF videos in (Acceptor/Donor), two enhanced TIFFs out.

### Step 2: Load Input Videos

- Drag & drop, browse file/folder, or paste path and press Enter.
- Single: one viewer. Dual: both viewers. Folders are batch processed and paired by relative path.

### Step 3: Select Model + Config

- Model: `.pth` checkpoint.
- Config: `.yaml/.yml` matching the model.
- Auto-detect works if YAML is in the same folder with the same base name.

### Step 4: Select Output Folder

- Default: `./outputs/DAWN_result_YYYYMMDD_HHMMSS/`
- Outputs:
  - Single: `<input>_enhanced.tif`
  - Dual: `<acceptor>_acceptor_enhanced.tif`, `<donor>_donor_enhanced.tif`

### Step 5: Set Linear Gain (Optional)

- Multiply input intensity (default 1.0).
- Use >1 for dim inputs, <1 for bright inputs.
- Test data brightness should be consistent with training data brightness.

### Step 6: Select Device

- Prefer GPU (CUDA). CPU is supported but very slow.
- System Monitor shows CPU/GPU usage during processing.

### Step 7: Start Processing

- Click **Reconstruct**. Use **Stop** to interrupt.
- Progress bar shows completion status.

### Step 8: View Results

- Outputs appear in the viewers and are saved in the output folder.

### Notes

- Batch inference supports nested folders and keeps Acceptor/Donor pairing by relative path.
- If errors appear, check input format and model/config compatibility.

## Updates:

### Ver. 1.1 (2026-03-10)
- Added batch inference with recursive folder search and relative-path pairing for dual-channel inputs.
- Added file/folder browsing support in input fields.
- Added Stop button to interrupt reconstruction.
- Updated dual-channel output naming to include _acceptor/_donor suffix.

### Ver. 1.0 (2026-01-11)
- Initial release.
