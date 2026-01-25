# MUFFLE GUI Manual

This is a GUI application for our paper **"DAWN: A Spatiotemporal Self-supervised Framework for Reconstructing Ultra–Low-Photon Single-Molecule Fluorescence Images"**.This method is compatible with both EMCCD and sCMOS detectors and extends seamlessly to live-cell imaging, where it reliably reconstructs molecular diffusion behavior even at extremely low photon flux. We aim to use this application to easily enhance the quality of videos captured by EMCCD or SCMOS cameras.And this application also supports MUFFLE model from **"Supervised multi-frame dual-channel denoising enables long-term single-molecule FRET under extremely low photon budget"**,which uses a supervised method, in contrast to our self-supervised approach.


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
   - In Progress

---


## Quick Start Guide

This guide will walk you through the complete workflow from loading videos to viewing enhanced results.

### Step 1: Select Processing Mode

Choose between two processing modes based on your input data:

- **Single Channel Mode**: 
  - Select this mode when you have a single grayscale video to enhance
  - Uses DBSN_centrblind model architecture
  - Input: One TIFF video file
  - Output: One enhanced video file
  
- **Dual Channel Mode**:
  - Select this mode when you have two correlated fluorescence videos (e.g., Acceptor and Donor channels)
  - Uses DBSN_fusion (for DAWN) or MUFFLE model architecture
  - Input: Two TIFF video files (Acceptor and Donor)
  - Output: Two enhanced video files

**How to select**: Click the radio button next to "Single Channel" or "Dual Channel" in the control panel on the right side of the interface.

### Step 2: Load Input Videos

Load your video files by drag-and-drop **or** by typing a file/folder path:

**For Single Channel Mode:**
1. Drag a TIFF video onto the "Single channel input" viewer **or** type a file/folder path in the input box and press Enter
2. The video will automatically load and display the first frame
3. Use the slider below the viewer to navigate through frames
4. If a folder path is provided, all TIFF files inside will be processed in batch (sorted by filename)

**For Dual Channel Mode:**
1. Drag a TIFF video onto the "Acceptor dark" viewer **or** type a file/folder path in the input box and press Enter
2. Drag a TIFF video onto the "Donor dark" viewer **or** type a file/folder path in the input box and press Enter
3. Both videos should load and display their first frames
4. Ensure both videos have the same number of frames and dimensions
5. If folder paths are provided, the two folders must contain the same number of TIFF files (paired by sorted filename)


### Step 3: Select Model and Configuration Files

Load the appropriate model and configuration files for your processing mode:

**Model Files:**
- **DAWN Models** (Self-supervised): Use `.pth` files
  - These are PyTorch model checkpoints from DAWN training
  - Example: `dbsn_gray_CT_data_ckpt_e39.pth`
  
- **MUFFLE Models** (Supervised): Use `.pkl` files
  - These are pickle-serialized model files from MUFFLE training
  - Example: `MUFFLE_model.pkl`

**Configuration Files:**
- YAML configuration files (`.yaml` or `.yml`)
- Must correspond to the model file (trained together)
- Contains model architecture parameters and settings

**How to Load:**

1. **Browse Model File**:
   - Click the "Browse model file" button
   - Navigate to your model file location
   - Select the `.pth` or `.pkl` file
   - Click "Open"
   - The model path will be displayed in the text area

2. **Browse Configuration File**:
   - Click the "Browse Config file (.yaml)" button
   - Navigate to your YAML configuration file
   - Select the `.yaml` or `.yml` file
   - Click "Open"
   - The config path will be displayed in the text area

3. **Auto-Detection** (Convenient):
   - When you load a model file, the GUI automatically searches for a matching YAML file
   - If a YAML file with the same base name is found in the same directory, it will be auto-selected
   - You can still manually change the YAML file if needed using the "Browse Config file" button

4. **Drag-and-Drop** (Alternative):
   - You can also drag and drop model files (`.pth` or `.pkl`) directly onto the model path display area
   - Auto-detection of YAML files will occur if available

**Important Notes:**
- Ensure the model file matches your selected processing mode (single channel models for single channel mode, dual channel models for dual channel mode)
- The configuration file must be compatible with the model file
- If you get errors about model file format, verify that the file extension matches the model type (`.pth` for DAWN, `.pkl` for MUFFLE)

### Step 4: Select Output Folder

Choose where to save the enhanced output videos:

1. **Default Location**: 
   - The default output folder is `./outputs` (relative to the application directory)
   - Outputs are automatically saved in timestamped subfolders
   - Format: `DAWN_result_YYYYMMDD_HHMMSS/`

2. **Custom Location**:
   - Click the "Browse Output Folder" button
   - Navigate to your desired output directory
   - Select the folder and click "OK"
   - The output path will be displayed in the text area

3. **Output Files**:
    - **Single Channel Mode**: `<input_name>_enhanced.tif` (16-bit TIFF sequence)
    - **Dual Channel Mode**: 
       - `<acceptor_name>_enhanced.tif` (16-bit TIFF sequence)
       - `<donor_name>_enhanced.tif` (16-bit TIFF sequence)
    - Batch runs generate one output file per input file
    - All outputs are saved as 16-bit TIFF files with values scaled to 0-65535

### Step 5: Set Linear Gain (Optional)

Adjust the input signal intensity before processing:

**What is Linear Gain?**
- A multiplicative factor applied to input video intensities
- Useful when there's a brightness mismatch between training data and test data
- Default value: 1.0 (no scaling)

**When to Adjust:**
- Increase gain (> 1.0) if your test videos are dimmer than training data
- Decrease gain (< 1.0) if your test videos are brighter/saturated compared to training data
- If brightness matches between training and test data, keep default value (1.0)

**How to Set:**
1. Enter the desired gain value in the "Linear Gain" input box (e.g., 1.5, 2.0, 0.8)
2. Click the "Set Linear Gain" button to apply
3. The input video previews will update to show the effect of the gain
4. Use the frame sliders to preview different frames with the applied gain

**Note**: Linear gain only affects the processing input, not the model architecture itself.

### Step 6: Select Processing Device

Choose between GPU (recommended) or CPU for processing:

**GPU (Recommended for Fast Processing):**
- **Requirements**: NVIDIA GPU with CUDA support
- **How to Select**:
  - In the device list, you'll see available GPUs as "GPU Name (cuda:0)", "GPU Name (cuda:1)", etc.
  - Click on the desired GPU device
  - The selected device will be highlighted

**CPU (Compatibility Option):**
- **Requirements**: No special hardware needed
- **Disadvantages**:
  - Extremely slow processing (not recommended for production use)
  - Warning dialog will appear if selected
- **How to Select**:
  - Click on "CPU (cpu)" in the device list
  - Confirm the warning dialog if it appears

**System Monitor**:
- The System Monitor panel shows real-time resource usage
- CPU usage, memory, GPU utilization, and temperature are displayed
- Use this to monitor processing and identify bottlenecks
- Monitoring starts automatically when the application launches

### Step 7: Start Processing

Begin the video enhancement process:

1. **Review Your Settings**:
   - Verify that all input videos are loaded correctly
   - Confirm model and config files are selected
   - Check that output folder is set appropriately
   - Ensure the correct processing mode is selected

2. **Click "Reconstruct" Button**:
   - Located at the bottom right of the control panel
   - Click once to start the processing
   - The button will be disabled during processing to prevent multiple runs
   - Click **"Stop"** to interrupt the current reconstruction if needed

3. **Monitor Progress**:
   - Watch the progress bar (0-100%) to track completion
   - Check the System Monitor for CPU/GPU usage and memory consumption
   - Processing time varies based on:
     - Video size (dimensions and number of frames)
     - Selected device (GPU is much faster)
     - Model complexity

4. **Wait for Completion**:
   - Do not close the application during processing
   - Processing may take several minutes to hours depending on video size
   - The application interface remains responsive during processing

### Step 8: View and Save Results

After processing completes:

1. **Automatic Display**:
   - Enhanced videos automatically appear in the output viewers
   - Single channel mode: "Single channel recon." viewer
   - Dual channel mode: "Acceptor recon." and "Donor recon." viewers

2. **Navigate Through Frames**:
   - Use the sliders below each output viewer to navigate through frames
   - Compare input and output videos side-by-side
   - Evaluate the enhancement quality frame by frame

3. **Save Results**:
   - Output files are automatically saved to the selected output folder
   - A success dialog will appear when processing completes
   - Click "Open folder" in the dialog to directly access the output files
   - Or manually navigate to the output folder

4. **Output File Locations**:
   - Navigate to: `[Output Folder]/DAWN_result_YYYYMMDD_HHMMSS/`
   - Find your enhanced video files (16-bit TIFF format)
   - Files are ready for further analysis or visualization

### Additional Features

#### Match Output Intensity (Dual Channel Mode Only)

When processing dual channel videos:
- Check the "Match Intensities of 2 Channels" checkbox before processing
- This normalizes the output intensities between Acceptor and Donor channels
- The Donor channel output will be scaled to match the Acceptor channel's mean intensity
- Useful for FRET analysis and visualization

### Note:

- **Error Handling:**
  - If an error occurs during the reconstruction process, an error message will be displayed. Please review the error message for details.
  - If you encounter an error, please check that the input videos are in the correct format and themodel file is valid.
  - If you are still unable to resolve the error, please contact us by adding an issue.

## Updates:

### Ver. 1.0 (2026-01-11)
- Initial release.
