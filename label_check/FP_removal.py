'''
    python FP_removal.py \
        -d=/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop \
        -r=/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled/preds \
        -s=/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled/QC \
        -p=/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/checklist_unlabeled_small.xlsx

    python FP_removal.py \
        -d=/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop \
        -r=/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled/preds \
        -s=/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled/QC \
        -p=/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/checklist_unlabeled_small.xlsx

'''
import pandas as pd
import numpy as np
import argparse
import cv2
import os

# the [x, y] for each right-click event will be stored here
right_clicks = []

def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == 2:
        global right_clicks

        # store the coordinates of the right-click event
        right_clicks.append([x, y])

        # this just verifies that the mouse data is being collected
        # you probably want to remove this later
        print(right_clicks)

def main(args):
    num_show = 25
    show_every = 1
    ck_path = args.ck_path
    checklist_df = pd.read_excel(ck_path, skiprows=1, dtype={"MRN": str, "date": str})
    checklist_df["MRN"] = checklist_df["MRN"].str.zfill(9)
    result_dir = args.result_dir
    data_dir = args.data_dir
    all_patients = [i for i in os.listdir(data_dir) if
                    os.path.isdir(os.path.join(data_dir, i)) and i[:4] == "Lung"]
    # namelist = np.load(os.path.join(result_dir, "namelist.npy"))
    for index, row in checklist_df.iterrows():
        mrn, dstr = row["MRN"], row["date"]
        identifier = "-".join([mrn, dstr])
        # assert identifier in namelist
        pred_dir = os.path.join(result_dir, identifier)
        pbb = np.load(os.path.join(pred_dir, "pbb.npy"))

        data_folder = [i for i in all_patients if i.endswith(mrn)][0]
        img_dir = os.path.join(data_dir, data_folder)
        img_path = os.path.join(img_dir, "{:s}.npz".format(identifier))
        try:
            image = np.load(img_path, allow_pickle=True)["image"]
        except:
            image = np.load(img_path.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
            image = image.squeeze(0)

        h, w = image.shape[1:]
        scale_width = 640 / w
        scale_height = 480 / h
        scale = min(scale_width, scale_height)
        window_width = int(w * scale)
        window_height = int(h * scale)
        cv2.namedWindow('slices', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('slices', window_width, window_height)
        #
        for pred in pbb:
            prob = round(1 / (1 + np.exp(-pred[0])), 5)
            z, y, x, d = pred[1:]
            start_with = int(z - num_show // 2 * show_every)
            ids = np.array([start_with + i*show_every for i in range(num_show)])
            # convert form RGB to BGR
            imgs = [cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR) for i in ids]
            # Display the gif
            i = 0
            while True:
                cv2.imshow("slices", imgs[i])
                if cv2.waitKey(100) & 0xFF == 27:
                    break
                i = (i + 1) % num_show
            cv2.destroyAllWindows()


        # bbox_images = [i for i in os.listdir(pred_dir) if i.endswith("_bbox.png")]
        # for bbox in bbox_images:
        #     path_image = os.path.join(pred_dir, bbox)
        #     img_name = identifier + "_" + bbox[:-4]
        #     img = cv2.imread(path_image, 1)
        #     scale_width = 640 / img.shape[1]
        #     scale_height = 480 / img.shape[0]
        #     scale = min(scale_width, scale_height)
        #     window_width = int(img.shape[1] * scale)
        #     window_height = int(img.shape[0] * scale)
        #     cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(img_name, window_width, window_height)
        #
        #     # set mouse callback function for window
        #     cv2.setMouseCallback(img_name, mouse_callback)
        #
        #     cv2.imshow(img_name, img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

    # # the [x, y] for each right-click event will be stored here
    # global right_clicks
    # right_clicks = list()
    #
    # # this function will be called whenever the mouse is right-clicked
    # def mouse_callback(event, x, y, flags, params):
    #     # right-click event value is 2
    #     if event == 2:
    #         global right_clicks
    #
    #         # store the coordinates of the right-click event
    #         right_clicks.append([x, y])
    #
    #         # this just verifies that the mouse data is being collected
    #         # you probably want to remove this later
    #         print(right_clicks)


    # path_image = "/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled/preds/031491426-20120712/pred_0_bbox.png"
    #
    # # path_image = os.path.join(pred_dir, "pred_0_bbox.png")
    # img = cv2.imread(path_image, 1)
    # scale_width = 640 / img.shape[1]
    # scale_height = 480 / img.shape[0]
    # scale = min(scale_width, scale_height)
    # window_width = int(img.shape[1] * scale)
    # window_height = int(img.shape[0] * scale)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', window_width, window_height)
    #
    # # set mouse callback function for window
    # cv2.setMouseCallback('image', mouse_callback)
    #
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





if __name__ == '__main__':

    # list_float_parser = lambda x: [float(i) for i in x.strip("[]").split(",")] if x else []
    parser = argparse.ArgumentParser(description="False positive nodule removal tool")
    parser.add_argument('-s', '--save_dir', type=str, help='save directory', default=None)
    parser.add_argument('-d', '--data_dir', type=str, help='data directory', default=None)
    parser.add_argument('-r', '--result_dir', type=str, help='result directory', default=None)
    parser.add_argument('-p', '--ck_path', type=str, help='checklist path', default=None)
    parser.add_argument('-a', '--annot_file', type=str, help='annotation file name', default="")
    parser.add_argument('-m', '--mask', type=eval, help='only apply mask in preprocessing', default=True)
    parser.add_argument('-c', '--crop', type=eval, help='crop masked images in preprocessing', default=True)
    parser.add_argument('-n', '--normalize', type=eval, help='normalized or raw data', default=True)
    args = parser.parse_args()
    main(args)