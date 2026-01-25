"""
DAWN Inference classes for DBSN models
Supports both single channel and dual channel inference
"""
import sys
import os
from os.path import join as opj
import json

import numpy as np
import torch
import logging as L

from Net.DBSN import DBSN_SingleChannel, DBSN_DualChannel

# #region agent log
# _log_file = r"c:\Users\Admin\Desktop\MUFFLE\Cursor_implement\.cursor\debug.log"
# def _log(hypothesis_id, location, message, data):
#     try:
#         with open(_log_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":hypothesis_id,"location":location,"message":message,"data":data,"timestamp":int(__import__("time").time()*1000)}, ensure_ascii=False)+"\n")
#     except: pass
# #endregion


class DAWN_SingleChannelInference(object):
    """Single channel inference using DBSN_centrblind model"""
    def __init__(self, opt, modelpath, device="cuda", imgsize=[200, 400]):
        self.imgsize = imgsize
        self.opt = opt
        self.opt.Img_size = imgsize
        L.info("CUDA Device Count: " + str(torch.cuda.device_count()))
        
        self.device = device
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:28", "创建模型前", {
        #     "Input_frame_num": getattr(opt, 'Input_frame_num', None),
        #     "Output_channel": getattr(opt, 'Output_channel', None),
        #     "Middle_channel": getattr(opt, 'Middle_channel', None),
        #     "Blindspot_conv_type": getattr(opt, 'Blindspot_conv_type', None),
        #     "Br1_block_num": getattr(opt, 'Br1_block_num', None),
        #     "Br2_block_num": getattr(opt, 'Br2_block_num', None)
        # })
        # #endregion
        
        self.net = DBSN_SingleChannel(opt, device)
        
        # Save initial model state
        model_dict_initial = self.net.state_dict()
        first_key_initial = list(model_dict_initial.keys())[0]
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:33", "模型创建后_初始权重", {
        #     "first_key": first_key_initial,
        #     "first_param_mean": float(model_dict_initial[first_key_initial].mean()),
        #     "first_param_std": float(model_dict_initial[first_key_initial].std()),
        #     "total_params": len(model_dict_initial)
        # })
        # #endregion
        
        # Load model weights (support both .pth and .pkl formats)
        model_file = torch.load(modelpath, map_location=self.device if "cpu" in device else torch.device(device))
        
        # Handle different checkpoint formats
        if 'state_dict_dbsn' in model_file:
            # PTH format from dbsn_gray training
            state_dict = model_file['state_dict_dbsn']
        elif 'state_dict' in model_file:
            state_dict = model_file['state_dict']
        elif isinstance(model_file, dict) and all(k.startswith('module.') or '.' in k for k in model_file.keys()):
            state_dict = model_file
        else:
            raise ValueError(f"Unknown checkpoint format in {modelpath}. Expected 'state_dict_dbsn', 'state_dict', or direct state_dict.")
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict_cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:50", "checkpoint_state_dict", {
        #     "state_dict_keys_count": len(state_dict_cleaned),
        #     "sample_keys": list(state_dict_cleaned.keys())[:5],
        #     "first_key": list(state_dict_cleaned.keys())[0] if state_dict_cleaned else None,
        #     "first_param_mean": float(list(state_dict_cleaned.values())[0].mean()) if state_dict_cleaned else None
        # })
        # #endregion
        
        # Filter out keys that don't exist in the model
        model_dict = self.net.state_dict()
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:55", "模型期望的键名", {
        #     "model_keys_count": len(model_dict),
        #     "sample_model_keys": list(model_dict.keys())[:5],
        #     "first_model_key": list(model_dict.keys())[0] if model_dict else None
        # })
        # #endregion
        
        # Check if checkpoint keys need 'net.' prefix
        sample_ckpt_key = list(state_dict_cleaned.keys())[0] if state_dict_cleaned else ""
        sample_model_key = list(model_dict.keys())[0] if model_dict else ""
        needs_net_prefix = (not sample_ckpt_key.startswith("net.")) and sample_model_key.startswith("net.")
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:60", "键名前缀检查", {
        #     "sample_ckpt_key": sample_ckpt_key,
        #     "sample_model_key": sample_model_key,
        #     "needs_net_prefix": needs_net_prefix
        # })
        # #endregion
        
        # Add 'net.' prefix if needed
        if needs_net_prefix:
            state_dict_cleaned = {f"net.{k}": v for k, v in state_dict_cleaned.items()}
            # #region agent log
            # _log("H", "DAWN_Inference.py:65", "添加net前缀后", {
            #     "sample_keys": list(state_dict_cleaned.keys())[:5]
            # })
            # #endregion
        
        pretrained_dict = {k: v for k, v in state_dict_cleaned.items() if k in model_dict}
        
        # Check if all required keys are present
        missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
        unexpected_keys = set(pretrained_dict.keys()) - set(model_dict.keys())
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:70", "键匹配结果", {
        #     "pretrained_keys_count": len(pretrained_dict),
        #     "missing_keys_count": len(missing_keys),
        #     "missing_keys_sample": list(missing_keys)[:5],
        #     "matched_keys_sample": list(pretrained_dict.keys())[:5]
        # })
        # #endregion
        
        if missing_keys:
            L.warning(f"Missing keys in checkpoint: {list(missing_keys)[:5]}...")
            print(f"WARNING: {len(missing_keys)} keys missing in checkpoint!")
        if unexpected_keys:
            L.warning(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:5]}...")
        
        # Load the state dict
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict, strict=False)
        
        # Check if weights changed after loading
        model_dict_loaded = self.net.state_dict()
        first_key = list(model_dict_loaded.keys())[0]
        weights_changed = not torch.allclose(model_dict_initial[first_key_initial], model_dict_loaded[first_key], atol=1e-6)
        
        # #region agent log
        # _log("H", "DAWN_Inference.py:85", "权重加载后检查", {
        #     "first_key": first_key,
        #     "initial_mean": float(model_dict_initial[first_key_initial].mean()),
        #     "loaded_mean": float(model_dict_loaded[first_key].mean()),
        #     "weights_changed": weights_changed,
        #     "sample_params": {k: float(v.mean()) for k, v in list(model_dict_loaded.items())[:3]}
        # })
        # #endregion
        
        if not weights_changed:
            print("ERROR: Model weights did NOT change after loading! Checkpoint may not be compatible.")
            # #region agent log
            # _log("H", "DAWN_Inference.py:90", "权重未改变_错误", {"weights_changed": False})
            # #endregion
        
        # Wrap with DataParallel if using GPU (after loading weights)
        if "cpu" not in device:
            self.net = torch.nn.DataParallel(self.net, device_ids=[int(device.split(':')[1])])
            self.net = self.net.to(device)
        
        self.net.eval()
        L.info(f"Single channel DBSN model loaded from {modelpath}!")
    
    def load_video(self, video_data):
        """
        Load single channel video data
        Args:
            video_data: numpy array of shape (frames, H, W) or (1, frames, 1, H, W)
        """
        if len(video_data.shape) == 3:
            # (frames, H, W)
            self.video_data = video_data
        elif len(video_data.shape) == 5:
            # (1, frames, 1, H, W) -> (frames, H, W)
            self.video_data = video_data[0, :, 0, :, :]
        else:
            raise ValueError(f"Unexpected video shape: {video_data.shape}")
    
    def gen_sliding_window(self, frames, fidx, window_size):
        """
        Generate sliding window for frame fidx
        Args:
            frames: numpy array (frames, H, W)
            fidx: current frame index
            window_size: number of frames in window
        Returns:
            window: numpy array (window_size, H, W)
        """
        half_num = window_size // 2
        fnum = frames.shape[0]
        
        if fidx < half_num:
            # Pad with first frame
            res = [frames[0:1]] * (half_num - fidx)
            res += [frames[: fidx + half_num + 1]]
        elif fidx >= fnum - half_num:
            # Pad with last frame
            res = [frames[fidx - half_num :]]
            res += [frames[fnum - 1 : fnum]] * (fidx - fnum + half_num + 1)
        else:
            # Normal case
            res = [frames[fidx - half_num : fidx + half_num + 1]]
        
        return np.concatenate(res, axis=0)
    
    def inference(self, gain=1, progress_recall=None):
        """
        Run inference on loaded video
        Args:
            gain: linear gain factor
            progress_recall: callback function for progress update
        Returns:
            output: numpy array (frames, H, W) - denoised video
        """
        vid_fnum = self.video_data.shape[0]
        window_size = getattr(self.opt, 'Input_frame_num', 3)
        
        outputs = []
        for fidx in range(vid_fnum):
            # Generate sliding window
            window = self.gen_sliding_window(self.video_data, fidx, window_size)
            
            # Convert to tensor: (window_size, H, W) -> (1, window_size, H, W)
            window_tensor = torch.from_numpy(window).to(self.device).float().unsqueeze(0)
            
            # Apply gain and clip
            window_tensor = torch.clamp(window_tensor * gain, 0, 1)
            
            # Inference
            with torch.no_grad():
                # #region agent log
                # if fidx < 3:  # Only log first 3 frames
                #     _log("I", "DAWN_Inference.py:145", "推理前_输入", {
                #         "frame_idx": fidx,
                #         "window_shape": list(window_tensor.shape),
                #         "window_min": float(window_tensor.min()),
                #         "window_max": float(window_tensor.max()),
                #         "window_mean": float(window_tensor.mean()),
                #         "window_std": float(window_tensor.std())
                #     })
                # #endregion
                
                output_raw = self.net(window_tensor)
                
                # #region agent log
                # if fidx < 3:  # Only log first 3 frames
                #     _log("I", "DAWN_Inference.py:150", "推理后_原始输出", {
                #         "frame_idx": fidx,
                #         "output_shape": list(output_raw.shape),
                #         "output_min": float(output_raw.min()),
                #         "output_max": float(output_raw.max()),
                #         "output_mean": float(output_raw.mean()),
                #         "output_std": float(output_raw.std()),
                #         "output_has_nan": bool(torch.isnan(output_raw).any()),
                #         "output_has_inf": bool(torch.isinf(output_raw).any())
                #     })
                # #endregion
                
                output = output_raw.clamp(0, 1)
                
                # #region agent log
                # if fidx < 3:  # Only log first 3 frames
                #     _log("I", "DAWN_Inference.py:155", "clamp后_输出", {
                #         "frame_idx": fidx,
                #         "output_min": float(output.min()),
                #         "output_max": float(output.max()),
                #         "output_mean": float(output.mean())
                #     })
                # # #endregion
                
                # print(f"window_tensor shape: {window_tensor.shape}||window_tensor min: {window_tensor.min()}||window_tensor max: {window_tensor.max()}||window_tensor mean: {window_tensor.mean()}")
                # print(f"output shape: {output.shape}||output min: {output.min()}||output max: {output.max()}||output mean: {output.mean()}")
            
            # Extract middle frame output: (1, 1, H, W) -> (H, W)
            output_frame = output[0, 0, :, :].detach().cpu().numpy()
            outputs.append(output_frame)
            
            
            if fidx % 100 == 0:
                print(f"Frame {fidx}/{vid_fnum} done.")
            if progress_recall is not None:
                progress_recall(10 + 90 * fidx / vid_fnum)
        
        return np.array(outputs)


class DAWN_DualChannelInference(object):
    """Dual channel inference using DBSN_fusion model"""
    def __init__(self, opt, modelpath, device="cuda", imgsize=[200, 400]):
        self.imgsize = imgsize
        self.opt = opt
        self.opt.Img_size = imgsize
        L.info("CUDA Device Count: " + str(torch.cuda.device_count()))
        
        self.device = device
        self.net = DBSN_DualChannel(opt, device)
        
        # Load model weights (support both .pth and .pkl formats)
        model_file = torch.load(modelpath, map_location=self.device if "cpu" in device else torch.device(device), weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict_dbsn' in model_file:
            # PTH format from dbsn_gray training
            state_dict = model_file['state_dict_dbsn']
        elif 'state_dict' in model_file:
            # Standard checkpoint format
            state_dict = model_file['state_dict']
        elif isinstance(model_file, dict) and all(k.startswith('module.') or '.' in k for k in model_file.keys()):
            # Direct state_dict format
            state_dict = model_file
        else:
            raise ValueError(f"Unknown checkpoint format in {modelpath}. Expected 'state_dict_dbsn', 'state_dict', or direct state_dict.")
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict_cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Filter out keys that don't exist in the model
        model_dict = self.net.state_dict()
        
        # Check if checkpoint keys need 'net.' prefix (same as single channel)
        sample_ckpt_key = list(state_dict_cleaned.keys())[0] if state_dict_cleaned else ""
        sample_model_key = list(model_dict.keys())[0] if model_dict else ""
        needs_net_prefix = (not sample_ckpt_key.startswith("net.")) and sample_model_key.startswith("net.")
        
        # Add 'net.' prefix if needed
        if needs_net_prefix:
            state_dict_cleaned = {f"net.{k}": v for k, v in state_dict_cleaned.items()}
            print(f"Added 'net.' prefix to checkpoint keys for dual channel model")
        
        pretrained_dict = {k: v for k, v in state_dict_cleaned.items() if k in model_dict}
        
        # Check if all required keys are present
        missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
        unexpected_keys = set(pretrained_dict.keys()) - set(model_dict.keys())
        
        if missing_keys:
            L.warning(f"Missing keys in checkpoint: {list(missing_keys)[:5]}...")
            print(f"WARNING: {len(missing_keys)} keys missing in checkpoint for dual channel!")
        if unexpected_keys:
            L.warning(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:5]}...")
        
        # Load the state dict
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict, strict=False)
        
        # Wrap with DataParallel if using GPU (after loading weights)
        if "cpu" not in device:
            self.net = torch.nn.DataParallel(self.net, device_ids=[int(device.split(':')[1])])
            self.net = self.net.to(device)
        
        self.net.eval()
        L.info(f"Dual channel DBSN model loaded from {modelpath}!")
    
    def load_video(self, ch_a, ch_b):
        """
        Load dual channel video data
        Args:
            ch_a: numpy array of shape (frames, H, W) or (1, frames, 1, H, W) - channel A
            ch_b: numpy array of shape (frames, H, W) or (1, frames, 1, H, W) - channel B
        """
        if len(ch_a.shape) == 3:
            self.ch_a_data = ch_a
            self.ch_b_data = ch_b
        elif len(ch_a.shape) == 5:
            self.ch_a_data = ch_a[0, :, 0, :, :]
            self.ch_b_data = ch_b[0, :, 0, :, :]
        else:
            raise ValueError(f"Unexpected video shape: {ch_a.shape}")
    
    def gen_sliding_window(self, frames, fidx, window_size):
        """
        Generate sliding window for frame fidx
        Args:
            frames: numpy array (frames, H, W)
            fidx: current frame index
            window_size: number of frames in window
        Returns:
            window: numpy array (window_size, H, W)
        """
        half_num = window_size // 2
        fnum = frames.shape[0]
        
        if fidx < half_num:
            # Pad with first frame
            res = [frames[0:1]] * (half_num - fidx)
            res += [frames[: fidx + half_num + 1]]
        elif fidx >= fnum - half_num:
            # Pad with last frame
            res = [frames[fidx - half_num :]]
            res += [frames[fnum - 1 : fnum]] * (fidx - fnum + half_num + 1)
        else:
            # Normal case
            res = [frames[fidx - half_num : fidx + half_num + 1]]
        
        return np.concatenate(res, axis=0)
    
    def inference(self, gain=1, progress_recall=None):
        """
        Run inference on loaded video
        Args:
            gain: linear gain factor
            progress_recall: callback function for progress update
        Returns:
            output_a: numpy array (frames, H, W) - denoised channel A
            output_b: numpy array (frames, H, W) - denoised channel B
        """
        vid_fnum = self.ch_a_data.shape[0]
        window_size = getattr(self.opt, 'Input_frame_num', 3)
        
        outputs_a = []
        outputs_b = []
        for fidx in range(vid_fnum):
            # Generate sliding windows for both channels
            window_a = self.gen_sliding_window(self.ch_a_data, fidx, window_size)
            window_b = self.gen_sliding_window(self.ch_b_data, fidx, window_size)
            
            # Convert to tensor: (window_size, H, W) -> (1, window_size, H, W)
            window_a_tensor = torch.from_numpy(window_a).to(self.device).float().unsqueeze(0)
            window_b_tensor = torch.from_numpy(window_b).to(self.device).float().unsqueeze(0)
            
            # Apply gain and normalize channels
            # if torch.mean(window_a_tensor) > 1e-5 and torch.mean(window_b_tensor) > 1e-5:
            #     a_gain = torch.mean(window_b_tensor) / torch.mean(window_a_tensor)
            # else:
            #     a_gain = 1
            a_gain = 1
            # print(f"a_gain: {a_gain},before a intensity: {window_a_tensor.min()}, {window_a_tensor.max()}, {window_a_tensor.mean()} \
            # before b intensity: {window_b_tensor.min()}, {window_b_tensor.max()}, {window_b_tensor.mean()}")
            window_a_tensor = torch.clamp(window_a_tensor * a_gain * gain, 0, 1)
            window_b_tensor = torch.clamp(window_b_tensor * gain, 0, 1)
            
            # Inference
            with torch.no_grad():
                #TODO
                output_b, output_a = self.net(window_b_tensor, window_a_tensor)
                output_a = output_a.clamp(0, 1)
                output_b = output_b.clamp(0, 1)
            # print(f"after a intensity: {output_a.min()}, {output_a.max()}, {output_a.mean()} \
            # after b intensity: {output_b.min()}, {output_b.max()}, {output_b.mean()}")
            
            # Extract middle frame output and reverse gain
            output_a_frame = (output_a[0, 0, :, :] / a_gain).detach().cpu().numpy()
            output_b_frame = output_b[0, 0, :, :].detach().cpu().numpy()
            outputs_a.append(output_a_frame)
            outputs_b.append(output_b_frame)
            
            if fidx % 100 == 0:
                print(f"Frame {fidx}/{vid_fnum} done.")
            if progress_recall is not None:
                progress_recall(10 + 90 * fidx / vid_fnum)
        
        return np.array(outputs_a), np.array(outputs_b)

