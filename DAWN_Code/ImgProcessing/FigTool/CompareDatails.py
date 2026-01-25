import matplotlib.pyplot as plt
import os, cv2

def compare_details(img_list, imgnames, detail_num=2, window_size=200, save_path="imgdetails"):
    pos_list = None

    if pos_list is None:
        fig = plt.figure
        plt.imshow(img_list[0])
        pos_list = plt.ginput(detail_num)
    print(pos_list)
    for i in range(len(img_list)):
        img = img_list[i]
        cv2.imwrite("." + save_path + "/img_%s.png"%(imgnames[i]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for j in range(len(pos_list)):
            pos = pos_list[j]
            detail_img = img[int(pos[1])-window_size//2 : int(pos[1])+window_size//2,
                             int(pos[0])-window_size//2 : int(pos[0])+window_size//2]
            detail_img = cv2.resize(detail_img, [window_size*5, window_size*5], interpolation=cv2.INTER_NEAREST)
            cv2.imwrite("." + save_path + "/img_%s_detail%d.png"%(imgnames[i], j), cv2.cvtColor(detail_img, cv2.COLOR_RGB2BGR))


def read_images_from_folder(path):
    imgs = []
    imgnames = []
    for imgname in os.listdir(path):
        if not imgname[0] == ".":
            imgnames.append(imgname)
            img = cv2.cvtColor(cv2.imread(os.path.join(path, imgname)), cv2.COLOR_BGR2RGB)
            imgs.append(img)
    return imgs, imgnames


def read_images(path_list):
    return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in path_list]

if __name__=="__main__":
    # imgs, imgnames = read_images_from_folder("./simu8")
    frame = "0005.png"
    vid = "real0"
    type = "1_40"
    img_path = ["./PID448133_" + type + "/" + vid + "/rgb_noisy/" + frame,
                "./PID448133_" + type + "/" + vid + "/rgb_pred/" + frame,
                "./PID3996911_" + type + "/" + vid + "/rgb_pred/" + frame,
                "./PID24424_" + type + "/" + vid + "/vbm4d/" + frame,
                "./PID448133_" + type + "/" + vid + "/rgb_pred_fastdvdnet/" + frame,
                "./PID24424_" + type + "/" + vid + "/videnn_denoise/" + frame]
    
    img_names = ["rgbnoisy",
                 "rgbpred",
                 "rgbprednonir",
                 "vbm4d",
                 "fastdvdnet",
                 "videnn"]
    
    
    imgs = read_images(img_path)
    
    compare_details(imgs, img_names, save_path=type + "_" + vid + "_" + frame + "_details")