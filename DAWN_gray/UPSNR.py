import torch
import numpy as np
import tifffile as tiff

def MSE(f, x):
    return np.mean((f-x)**2)
    
def uMSE(f, a, b, c):
        # 转为 float 防止 uint 下溢
    f = f.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)
    return np.mean((a-f)**2) - np.mean((b-c)**2)/4


def uPSNR(f, a, b, c, max_val=1.0):
    mse_value = uMSE(f, a, b, c)
    if mse_value <= 0:
        print('x')
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse_value)


def read_tif(path1,path2):
    f_v = tiff.imread(path1)
    nos_v = tiff.imread(path2)
    # 判断位深
    if f_v.dtype == 'uint8':
        max_v = 255
    elif f_v.dtype == 'uint16':
        max_v = 65535
    f_v = f_v.astype(np.float32)
    nos_v = nos_v.astype(np.float32)

    return f_v, nos_v, max_v


def uPSNR_video(f_video,Nos_video,max_value = 65535):
    upsnr_list = []
    for i in range(f_video.shape[0]-2):
        uP = uPSNR(f_video[i],Nos_video[i],Nos_video[i+1],Nos_video[i+2],max_val=max_value)
        # print('Frame %d uPSNR: %.4f dB'%(i+1,uP))
        upsnr_list.append(uP)
        
    return np.mean(upsnr_list)

if __name__ == '__main__':
    # f_path = '/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_dbsn3DCB_Dark1022NA_31_0Kacc_AP2_47/s1203_0001_1.tif'
    # nos_path= '/hdd/0/lsy/EMCCD_dataset/20251022_Asplit/acc/800f/s1203_0001/s1203_0001_1.tif'
    f_path = '/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_dbsn3DCB_Dark1104SWNA_31_acc25_AP2_39/s1203_0000_1.tif'
    nos_path= '/hdd/0/lsy/EMCCD_dataset/20251104_Asplit/acc/800f/25%/s1203_0000/s1203_0000_1.tif'
    # f_path = '/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_dbsn3DCB_Dark915NA_31_0Kacc3.4_30/S1208_0003_1.tif'
    # nos_path= '/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/3.4%/acc/S1208_0003/S1208_0003_1.tif'
    # f_path = '/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_dbsn3DCB_Dark915NA_31_0Kacc6.6_29/S1207_0001_1.tif'
    # nos_path= '/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/6.6%/acc/S1207_0001/S1207_0001_1.tif'
    # f_path = '/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_dbsn3DCB_SCMOS1022NA_31_0acc1205_AP2_49/s1205_0000_1_1.tif'
    # nos_path= '/hdd/0/lsy/sCMOS_dataset/20251022_sCMOS_Split/acc/1205/s1205_0000_1/s1205_0000_1_1.tif'

    f_v, nos_v, max_v = read_tif(f_path,
                                 nos_path)
    upsnr = uPSNR_video(f_v,nos_v,max_value=max_v)
    print('uPSNR: %.4f dB'%upsnr)

    
    
    