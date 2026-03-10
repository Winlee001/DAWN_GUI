# 数据对齐
## 1.准备工作
下载PreproccessDataset.py、Calibration.py、data_test_pro.py文件，放入同一个文件夹
用户只需要修改data_test_pro.py文件，另外两个不需要修改
## 2.修改参数
### 基本参数
打开data_test_pro.py文件，修改主程序中的下列参数
```python
path_list = [
	's1105_0000',
	's1109_0000',
	's1205_0000',
	's1209_0000']
dark_list = ['s1205_0000', 's1209_0000']
bright_list = ['s1105_0000', 's1109_0000']
base_path = f"/data4/zsq/20241021_NoiseData4train"
base_save = '/ssd/1/zby/dataset/20241021_NoiseData4train_all'
```
base_path为存放待配准数据的文件夹地址
base_save为将要保存到的已配准数据的文件夹地址
path_list为本次要处理的数据种类的文件夹名称
dark_list存放将打上dark标签的数据的文件夹名称
bright_list存放将打上bright标签的数据的文件夹名称
### 可选参数
如需裁剪图像，需要调整下面的参数
```python
all_data = robust_calc(
            sample_file_glob=path,
            # save_path=f"DataPreprocess/outputs/Data20240506-sCMOS_{iden:s}",
            save_path=f"/ssd/1/zby/",
            # chal_roi=[250, 1900, 0, 900, ],
            # chbl_original=[0, 2048, 900, 2048, ],
            # chal_roi=[128, 384, 0, 256, ],
            # chbl_original=[128, 384, 256, 512, ],
            chal_roi=[0, 512, 0, 256, ],
            chbl_original=[0, 512, 256, 512, ],
        )  # chal_roi, chbl_original裁剪区间
```
chal_roi=[0, 512, 0, 256, ]代表如何裁剪左通道数据，纵轴0-512，横轴0-256
chbl_original=[0, 512, 256, 512, ]代表如何裁剪右通道数据，纵轴0-512，横轴256-512

```python
img1 = img[0, 0:512, 0:256]
img2 = img[0, 0:512, 256:512]
```
img[0, 0:512, 0:256]代表如何裁剪左通道数据，纵轴0-512，横轴0-256，保持与前面一致
img[0, 0:512, 256:512]代表如何裁剪右通道数据，纵轴0-512，横轴256-512，保持与前面一致
```python
res_img = img[j, 0:512, 256:512]
res_img_h = cv2.warpPerspective(res_img, H, (res_img.shape[1], res_img.shape[0]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)  # 透视变换 H*target = img2,img2经变换变成target域对应的imgout
res_img = img[j, 0:512, 0:256]
```
img[j, 0:512, 0:256]代表如何裁剪左通道数据，纵轴0-512，横轴0-256，保持与前面一致
img[j, 0:512, 256:512]代表如何裁剪右通道数据，纵轴0-512，横轴256-512，保持与前面一致
## 程序运行
```shell
python data_test_pro.py
```
## 可视化调试
打开PreproccessDataset.py文件，修改主程序中的下列参数
```python
base_path = f"/data4/zsq/20241021_NoiseData4train/s1101_0000/*.tif"
test_path = f"/data4/zsq/20241021_NoiseData4train/s1101_0000/s1101_0000.tif"
all_data = robust_calc(
	sample_file_glob=base_path,
	save_path=f"/ssd/1/zby/",
	chal_roi=[128, 384, 0, 256, ],
	chbl_original=[128, 384, 256, 512, ],
)  # chal_roi, chbl_original裁剪区间
print(all_data)
H = np.array(all_data["chbl_h_mat"])

img = tifffile.imread(test_path)
img1 = img[0, 128:384, 0:256]
img2 = img[0, 128:384, 256:512]
```
base_path为需要测试的具体文件夹目录，*.tif无需修改
test_path为需要测试的具体文件目录
save_path保存中途测试数据的地址
chal_roi=[128, 384, 0, 256,]代表如何裁剪左通道数据，纵轴128-384，横轴0-256
chbl_original=[128, 384, 256, 512,  ]代表如何裁剪右通道数据，纵轴128-51384，横轴256-512
img[0, 128:384, 0:256]代表如何裁剪左通道数据，纵轴128-384，横轴0-256，保持与前面一致，第一维度0代表查看第0帧
img[0, 128:384, 256:512]代表如何裁剪右通道数据，纵轴128-51384，横轴256-512，保持与前面一致，第一维度0代表查看第0帧
