import sys, os
import numpy as np
import cv2
import skimage.metrics as Skim
import sklearn.metrics as Sklm
import matplotlib.pyplot as plt
from tqdm import trange

def rgbnir_ssim(r, g, b, nir):
    return [
        Skim.structural_similarity(r, g),
        Skim.structural_similarity(r, b),
        Skim.structural_similarity(g, b),
        Skim.structural_similarity(r, nir),
        Skim.structural_similarity(g, nir),
        Skim.structural_similarity(b, nir)
        ]

def rgbnir_nmi(r, g, b, nir):
    r = r.reshape(-1)
    g = g.reshape(-1)
    b = b.reshape(-1)
    nir = nir.reshape(-1)
    return [
        Sklm.normalized_mutual_info_score(r, g),
        Sklm.normalized_mutual_info_score(r, b),
        Sklm.normalized_mutual_info_score(g, b),
        Sklm.normalized_mutual_info_score(r, nir),
        Sklm.normalized_mutual_info_score(g, nir),
        Sklm.normalized_mutual_info_score(b, nir)
        ]

if __name__ == "__main__":
    np.random.seed(0)

    DATA_PATH = "D:/proj-files-level2/dual-channel-low-light/dataset/rgb_nir_video_dataset_full_v4/train/"
    CALC_NUM = 50

    dir_list = os.listdir(DATA_PATH)
    ssim_list = []
    nmi_list = []
    for _ in trange(CALC_NUM):
        random_rgb_path = ""
        random_nir_path = ""
        while not (os.path.exists(random_rgb_path) and os.path.exists(random_nir_path)):
            random_dir = os.path.join(DATA_PATH, dir_list[int(np.random.random() * len(dir_list))])
            random_fname = np.random.choice(os.listdir(os.path.join(random_dir, "rgb")))
            random_rgb_path = os.path.join(random_dir, "rgb", random_fname)
            random_nir_path = os.path.join(random_dir, "nir", random_fname)

        rgb = cv2.cvtColor(cv2.imread(random_rgb_path), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (100, 100), interpolation=cv2.INTER_NEAREST)
        nirimg = cv2.cvtColor(cv2.imread(random_nir_path), cv2.COLOR_BGR2GRAY)
        nirimg = cv2.resize(nirimg, (100, 100), interpolation=cv2.INTER_NEAREST)
        rimg = rgb[:,:,0]
        gimg = rgb[:,:,1]
        bimg = rgb[:,:,2]
        ssim_list.append(rgbnir_ssim(rimg, gimg, bimg, nirimg))
        nmi_list.append(rgbnir_nmi(rimg, gimg, bimg, nirimg))


    scatter_size = 30

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[0], nmi[0], c="darkorange", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("R-G", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_rg.pdf"))
    plt.show()

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[1], nmi[1], c="darkmagenta", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("R-B", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_rb.pdf"))
    plt.show()

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[2], nmi[2], c="darkturquoise", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("G-B", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_gb.pdf"))
    plt.show()

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[3], nmi[3], c="tomato", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("R-NIR", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_rn.pdf"))
    plt.show()

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[4], nmi[4], c="yellowgreen", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("G-NIR", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_gn.pdf"))
    plt.show()

    plt.figure(figsize=[4,4])
    for ssim, nmi in zip(ssim_list, nmi_list):
        plt.scatter(ssim[5], nmi[5], c="dodgerblue", s=scatter_size)
    plt.xlabel("SSIM", fontsize=15)
    plt.ylabel("NMI", fontsize=15)
    plt.xlim([0,1])
    plt.ylim([0,0.6])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("B-NIR", fontsize=15)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),"corr_bn.pdf"))
    plt.show()