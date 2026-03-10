
# DAWN_gray Usage Instructions

This is the source code for **DAWN**.

---

## A. Data Preprocessing

1. We provide preprocessing scripts in the repository, including functionalities such as video frame splitting and brightness adjustment. You can refer to these scripts to prepare your own data. In our paper, for EMCCD data, we achieved optimal denoising results by splitting videos into three-frame inputs and sufficiently increasing brightness. **Make sure that the `input_channel` parameter in training matches the number of frames in your training data.**

2. For dual-channel mode, the data must be properly paired. For example, for donor and acceptor channels, the videos must have exactly the same dimensions (number of frames, height, width, and number of channels). Additionally, the positions of corresponding molecules in the frames should be as consistent as possible. If there is misalignment, we recommend using image registration methods. We provide related scripts for registration in the `/data_aliase` directory, which you can use to align your data.

3. After preprocessing, place your processed data paths and corresponding names into the `rootlist` of `image_dataset`. This ensures that the code correctly loads the matching datasets.

---

## B. Model Training and Testing Scripts

* **Single-channel:**

  * Training: `gray_unpair_pretrain_mu.py`
  * Inference: `gray_slidewindow_test.py`

* **Dual-channel:**

  * Training: `gray_dualchan_pretrain_mu.py`
  * Inference: `gray_slidwin_dual_test.py`

Remember to set your own save path in gray_slidewindow_test.py and gray_slidwin_dual_test.py.
---

## C. Example Scripts

We provide example scripts for training and testing in the repository:

* `gdatav4_dual.sh`
* `gray_dual_train.sh`

These scripts demonstrate how to call the training and testing code and can be used as a reference.
