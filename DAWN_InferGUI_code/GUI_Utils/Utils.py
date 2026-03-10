import sys
import os

from os.path import join as opj
from os.path import dirname as opd

import tifffile
import numpy as np
import time
from omegaconf import OmegaConf
import torch

from Test import DualChannelLowLight
from DAWN_Inference import DAWN_SingleChannelInference, DAWN_DualChannelInference

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import time
import sys


class ReconstructThread(QThread):
    resultReady = pyqtSignal(tuple)
    updateProgress = pyqtSignal(int)
    error = pyqtSignal(str)
    canceled = pyqtSignal(str)

    def __init__(self, parent, opt_path, model_path, mode, ch_a_numpy, ch_b_numpy, single_ch_numpy, 
                 selected_device, enhance_progress, output_path, linear_gain, input_paths_a=None,
                 input_paths_b=None, input_paths_single=None, enhance_a=True, enhance_b=True):
        super().__init__(parent=parent)
        self.opt_path = opt_path
        self.model_path = model_path
        self.mode = mode  # "single" or "dual"
        self.ch_a_numpy = ch_a_numpy
        self.ch_b_numpy = ch_b_numpy
        self.single_ch_numpy = single_ch_numpy
        self.selected_device = selected_device
        self.enhance_progress = enhance_progress
        self.output_path = output_path
        self.linear_gain = linear_gain
        self.input_paths_a = input_paths_a or []
        self.input_paths_b = input_paths_b or []
        self.input_paths_single = input_paths_single or []
        self.enhance_a = enhance_a
        self.enhance_b = enhance_b
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        

    def run(self):
        try:
            print(f"Reconstruct videos with device: {self.selected_device}, mode: {self.mode}")
            self.updateProgress.emit(5)
            if not self.selected_device == "cpu":
                if ":" in self.selected_device:
                    torch.cuda.set_device(int(self.selected_device.split(':')[1]))
                else:
                    torch.cuda.set_device(0)
            
            opt = OmegaConf.load(self.opt_path)
            self.updateProgress.emit(10)
            
            def progress_callback(file_index, total_files):
                def _callback(value):
                    if self._stop_requested:
                        raise InterruptedError("Reconstruction interrupted")
                    overall = ((file_index + value / 100) / max(total_files, 1)) * 100
                    self.updateProgress.emit(int(overall))
                return _callback

            def build_output_name(input_path, fallback_name, suffix=None):
                if input_path:
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    if suffix:
                        return f"{base_name}_{suffix}_enhanced.tif"
                    return f"{base_name}_enhanced.tif"
                return fallback_name

            if self.mode == "single":
                # Single channel mode
                if self.single_ch_numpy is None and not self.input_paths_single:
                    raise ValueError("Single channel data is required for single channel mode")
                
                # Check if using DBSN model (based on file extension and network name)
                network_name = getattr(opt, 'Network_name', '')
                is_pth_file = self.model_path.endswith('.pth')
                is_dbsn_model = 'DBSN' in network_name or is_pth_file
                
                time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                result_dir = opj(self.output_path, f"DAWN_result_{time_str}")
                os.makedirs(result_dir, exist_ok=True)

                input_paths = self.input_paths_single
                total_files = len(input_paths) if input_paths else 1
                last_output = None

                if input_paths:
                    for idx, input_path in enumerate(input_paths):
                        if self._stop_requested:
                            raise InterruptedError("Reconstruction interrupted")
                        data = load_tiff_seq(input_path) * self.linear_gain
                        if is_dbsn_model:
                            model = DAWN_SingleChannelInference(
                                opt, self.model_path, device=self.selected_device, 
                                imgsize=data.shape[1:]
                            )
                            model.load_video(data)
                            output = model.inference(gain=self.linear_gain, progress_recall=progress_callback(idx, total_files))
                            single_out_numpy = output
                        else:
                            model = DualChannelLowLight(opt, self.model_path, device=self.selected_device, 
                                                      imgsize=data.shape[1:])
                            in_single = data[None, :, None, :, :]
                            model.load_low_light_video(in_single, in_single)
                            out_a, out_b = model.start_nn(progress_recall=progress_callback(idx, total_files))
                            single_out_numpy = out_a[:, 0, :, :]

                        save_name = build_output_name(input_path, "Single_channel_enhanced.tif")
                        single_save = np.clip(single_out_numpy * 65535, 0, 65535).astype(np.uint16)
                        tifffile.imwrite(opj(result_dir, save_name), single_save)
                        last_output = single_out_numpy
                else:
                    data = self.single_ch_numpy
                    if is_dbsn_model:
                        model = DAWN_SingleChannelInference(
                            opt, self.model_path, device=self.selected_device, 
                            imgsize=data.shape[1:]
                        )
                        model.load_video(data)
                        output = model.inference(gain=self.linear_gain, progress_recall=progress_callback(0, total_files))
                        single_out_numpy = output
                    else:
                        model = DualChannelLowLight(opt, self.model_path, device=self.selected_device, 
                                                  imgsize=data.shape[1:])
                        in_single = data[None, :, None, :, :]
                        model.load_low_light_video(in_single, in_single)
                        out_a, out_b = model.start_nn(progress_recall=progress_callback(0, total_files))
                        single_out_numpy = out_a[:, 0, :, :]
                    save_name = build_output_name(None, "Single_channel_enhanced.tif")
                    single_save = np.clip(single_out_numpy * 65535, 0, 65535).astype(np.uint16)
                    tifffile.imwrite(opj(result_dir, save_name), single_save)
                    last_output = single_out_numpy

                self.updateProgress.emit(100)
                self.resultReady.emit((last_output, None, result_dir))
                
            else:
                # Dual channel mode
                if (self.ch_a_numpy is None or self.ch_b_numpy is None) and (not self.input_paths_a or not self.input_paths_b):
                    raise ValueError("Both channel A and B data are required for dual channel mode")
                
                # Check if using DBSN model (based on file extension and network name)
                network_name = getattr(opt, 'Network_name', '')
                is_pth_file = self.model_path.endswith('.pth')
                is_dbsn_model = 'DBSN' in network_name or is_pth_file

                time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                result_dir = opj(self.output_path, f"DAWN_result_{time_str}")
                os.makedirs(result_dir, exist_ok=True)

                input_paths_a = self.input_paths_a
                input_paths_b = self.input_paths_b
                total_files = len(input_paths_a) if input_paths_a else 1
                last_a_out = None
                last_b_out = None

                if input_paths_a and input_paths_b:
                    if len(input_paths_a) != len(input_paths_b):
                        raise ValueError("Acceptor and Donor input folders must contain the same number of TIFF files")
                    for idx, (path_a, path_b) in enumerate(zip(input_paths_a, input_paths_b)):
                        if self._stop_requested:
                            raise InterruptedError("Reconstruction interrupted")
                        data_a = load_tiff_seq(path_a) * self.linear_gain * self.enhance_a
                        data_b = load_tiff_seq(path_b) * self.linear_gain * self.enhance_b
                        if is_dbsn_model:
                            model = DAWN_DualChannelInference(
                                opt, self.model_path, device=self.selected_device,
                                imgsize=data_a.shape[1:]
                            )
                            model.load_video(data_a, data_b)
                            out_a, out_b = model.inference(gain=self.linear_gain, progress_recall=progress_callback(idx, total_files))
                            ch_a_out_numpy = out_a
                            ch_b_out_numpy = out_b
                        else:
                            model = DualChannelLowLight(opt, self.model_path, device=self.selected_device, 
                                                      imgsize=data_a.shape[1:])
                            in_a, in_b = data_a[None, :, None, :, :], data_b[None, :, None, :, :]
                            print(f"in_a.shape: {in_a.shape}. in_a mean: {np.mean(in_a)}, in_b.shape: {in_b.shape}. in_b mean: {np.mean(in_b)}")
                            model.load_low_light_video(in_a, in_b)
                            out_a, out_b = model.start_nn(progress_recall=progress_callback(idx, total_files))
                            ch_a_out_numpy = out_a[:, 0, :, :]
                            ch_b_out_numpy = out_b[:, 0, :, :]

                        save_name_a = build_output_name(path_a, "Acceptor_enhanced.tif", "acceptor")
                        save_name_b = build_output_name(path_b, "Donor_enhanced.tif", "donor")
                        ch_a_save = np.clip(ch_a_out_numpy * 65535, 0, 65535).astype(np.uint16)
                        ch_b_save = np.clip(ch_b_out_numpy * 65535, 0, 65535).astype(np.uint16)
                        tifffile.imwrite(opj(result_dir, save_name_a), ch_a_save)
                        tifffile.imwrite(opj(result_dir, save_name_b), ch_b_save)
                        last_a_out = ch_a_out_numpy
                        last_b_out = ch_b_out_numpy
                else:
                    data_a = self.ch_a_numpy
                    data_b = self.ch_b_numpy
                    if is_dbsn_model:
                        model = DAWN_DualChannelInference(
                            opt, self.model_path, device=self.selected_device,
                            imgsize=data_a.shape[1:]
                        )
                        model.load_video(data_a, data_b)
                        out_a, out_b = model.inference(gain=self.linear_gain, progress_recall=progress_callback(0, total_files))
                        ch_a_out_numpy = out_a
                        ch_b_out_numpy = out_b
                    else:
                        model = DualChannelLowLight(opt, self.model_path, device=self.selected_device, 
                                                  imgsize=data_a.shape[1:])
                        in_a, in_b = data_a[None, :, None, :, :], data_b[None, :, None, :, :]
                        print(f"in_a.shape: {in_a.shape}. in_a mean: {np.mean(in_a)}, in_b.shape: {in_b.shape}. in_b mean: {np.mean(in_b)}")
                        model.load_low_light_video(in_a, in_b)
                        out_a, out_b = model.start_nn(progress_recall=progress_callback(0, total_files))
                        ch_a_out_numpy = out_a[:, 0, :, :]
                        ch_b_out_numpy = out_b[:, 0, :, :]

                    save_name_a = build_output_name(None, "Acceptor_enhanced.tif", "acceptor")
                    save_name_b = build_output_name(None, "Donor_enhanced.tif", "donor")
                    ch_a_save = np.clip(ch_a_out_numpy * 65535, 0, 65535).astype(np.uint16)
                    ch_b_save = np.clip(ch_b_out_numpy * 65535, 0, 65535).astype(np.uint16)
                    tifffile.imwrite(opj(result_dir, save_name_a), ch_a_save)
                    tifffile.imwrite(opj(result_dir, save_name_b), ch_b_save)
                    last_a_out = ch_a_out_numpy
                    last_b_out = ch_b_out_numpy
                
                self.updateProgress.emit(100)
                self.resultReady.emit((last_a_out, last_b_out, result_dir))
                
        except InterruptedError as e:
            self.updateProgress.emit(0)
            self.canceled.emit(str(e))
        except Exception as e:
            import traceback
            traceback_info = traceback.format_exc()
            self.error.emit(str(e) + "\n" + str(traceback_info))
            print(e)
            print(traceback_info)




def load_tiff_seq(path):
    """Load a sequence of TIFF files from a given directory path.

    Args:
        path (str): Path to the directory containing the TIFF files.

    Returns:
        np.ndarray: A 3D array containing the TIFF files.
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        files.sort()
        files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        files = [opj(path, f) for f in files]
        img = tifffile.imread(files)
    else:
        img = tifffile.imread(path)
        
    # 标准化到0-1，根据数据类型确定最大值
    if img.dtype == np.uint8:
        img = img / 255
    elif img.dtype == np.uint16:
        img = img / 65535
    else:
        raise ValueError("Unknown data type: {}".format(img.dtype))
    
    if len(img.shape) == 4:
        if img.shape[3] == 1:
            img = img[:, :, :, 0]
        else:
            raise ValueError("The input image is not a 3D image.")
    
    return img