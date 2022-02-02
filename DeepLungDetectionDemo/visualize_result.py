from tqdm import tqdm
import matplotlib
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_bbox(savedir, images, pred, label=None, show=True, title=None, fontsize=10):
    '''
    plot center image with bbox
    :param images: CT scan, shape: (num_slices, h, w) or (h, w)
    :param label: coordinates & diameter (all in pixel space): (z, y, x, d) or (y, x, d)
    :param savedir: save directory
    :return: None
    '''
    fig, ax = plt.subplots(1)
    if label is not None and pred is not None:
        zp, yp, xp, dp = pred
        zl, yl, xl, dl = label
        ax.imshow(images[int(zl)], cmap="gray", vmin=0, vmax=255)
        rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                       facecolor='none')
        ax.add_patch(rect_label)
        if np.abs(zp - zl) <= dp:
            rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                          facecolor='none')
            ax.add_patch(rect_pred)
    elif label is not None:
        zl, yl, xl, dl = label
        ax.imshow(images[int(zl)], cmap="gray", vmin=0, vmax=255)
        rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                       facecolor='none')
        ax.add_patch(rect_label)
    elif pred is not None:
        zp, yp, xp, dp = pred
        ax.imshow(images[int(zp)], cmap="gray", vmin=0, vmax=255)
        rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                      facecolor='none')
        ax.add_patch(rect_pred)
    else:
        print("no prediction or label is given!")
    if title:
        plt.title(title, fontsize=fontsize)
    if show:
        plt.show()
    else:
        plt.savefig(savedir + "_bbox.png")
        plt.close()

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)): overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def nms(output, nms_th):
    if len(output) == 0: return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1: bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def show_single_result(srsid):
    pid, date = srsid.split("-")
    filepath = os.path.join(data_dir, "Lung_{:s}/{:s}_clean.npz".format(pid, srsid))
    ctdat = np.load(filepath, allow_pickle=True)["image"]
    ctlab = np.load(filepath.replace("clean", "label"), allow_pickle=True)["label"]

    print('scan name: ', srsid)
    print("image shape is: ", ctdat.shape)
    print("label shape is: ", ctlab.shape)
    # print(ctdat.shape, ctlab.shape)

    # plt.rcParams.update({'font.size': 5})

    for idx in range(ctlab.shape[0]):
        if abs(ctlab[idx, 0]) + abs(ctlab[idx, 1]) + abs(ctlab[idx, 2]) + abs(ctlab[idx, 3]) == 0: continue
        title = "series {:s} \n label {:s}".format(srsid, str(ctlab[idx]))
        save_path = os.path.join(save_dir, title)
        plot_bbox(save_path, ctdat[0], None, ctlab[idx], show=show, title=title, fontsize=10)
    plt.show()

    pbb = np.load(result_dir + srsid + '_pbb.npy')
    lbb = np.load(result_dir + srsid + '_lbb.npy')

    pbb = np.array(pbb[pbb[:, 0] > 0])
    pbb = nms(pbb, 0.1)

    delete_row = []
    for row, pi in enumerate(pbb):
        if np.any(pi[1:4] >= ctdat[0].shape):
            delete_row.append(row)
    pbb = np.delete(pbb, delete_row, 0)
    print("prediction shape is: ", pbb.shape)

    num_show = np.min([pbb.shape[0], 5])

    # print pbb.shape, pbb
    print('Detection Results according to confidence')
    for idx in range(num_show):
        pbb[idx, 0] = 1 / (1 + np.exp(-pbb[idx, 0]))
        title = "series {:s} \n predict {:s}".format(srsid, str(pbb[idx]))
        save_path = os.path.join(save_dir, title)
        plot_bbox(save_path, ctdat[0], pbb[idx][1:], None, title=title, show=show, fontsize=10)
    plt.show()

def show_results(srslst, showid):
    if srslst is None:
        srslst = np.load(os.path.join(result_dir, "namelist.npy")).tolist()
    if showid is not None:
        show_single_result(srslst[showid])
    else:
        for showid in tqdm(range(len(srslst))):
            srsid = srslst[showid]
            show_single_result(srsid)


if __name__ == '__main__':
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    # data_dir = "/data/pyuan2/Methodist_incidental/data_kim/labeled/"
    # data_dir = "/data/pyuan2/Methodist_incidental/data_kim/masked_first/"
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/masked_with_crop/"
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug/"
    # data_dir = "./Methodist_incidental/data_Ben/modeNorm3/"

    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/labeled/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210121-225702/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210121-180624/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210209-104946/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210209-122426-test/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector/results/res18-20210126-011543/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/worker32_batch8_kim_masked_PET/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results_zshift/worker32_batch8_kim_masked_crop_nonPET_lr001_rs128_augNone/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_mask_crop_nonPET_lr001_rs128_limit1.0/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_det/ection_pytorch/detector_ben/results/worker32_batch8_kim_mask_crop_nonPET_lr001_rs128_augAll_limit1.0/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_ben_nonPET_lr001_rs42_limit1.0_5fold_0/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs42_5fold_0/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs42_augAll_5fold_0/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs42_5fold_4/bbox/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210429-200058-train/bbox/"


    # result_dir = "./detector_ben/results/methodist_finetuned_mode3/bbox/"
    # result_dir = "./detector_ben/results/methodist_pretrainedLUNA_mode3/bbox"

    # data_dir = "./Methodist_incidental/data_Ben/preprocessed_data_v1/"
    # result_dir = "./detector_ben/results/methodist_finetuned_minmax_newLungSeg/bbox/"

    # data_dir = "./Methodist_incidental/data_Ben/preprocessed/"
    # result_dir = "./detector_ben/results/methodist_finetuned_minmax/bbox/"

    # ## Old mask
    # data_dir = "./Methodist_incidental/data_Ben/preprocessed/"
    # result_dir = "./detector_ben/results/methodist_finetuned_minmax_fixTest/bbox/"

    ## New mask
    data_dir = "./Methodist_incidental/data_Ben/preprocessed_data_v1/"
    result_dir = "./detector_ben/results/methodist_finetuned_minmax_newLungSeg_fixTest/bbox/"

    save_dir = result_dir.replace("bbox", "images")
    # srslst = ["patient018-20121001"]
    srslst = None
    # showid = 0
    showid = None
    show = False
    os.makedirs(save_dir, exist_ok=True)
    show_results(srslst, showid)

    # # ## examples in demo
    # srslst = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.208737629504245244513001631764',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.108231420525711026834210228428',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.161002239822118346732951898613',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.162901839201654862079549658100',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.179162671133894061547290922949']
    #
    # data_dir = "CT/"
    # result_dir = "detection/"

    ## luna detection results
    # srslst = ["1.3.6.1.4.1.14519.5.2.1.6279.6001.121108220866971173712229588402",\
    #           "1.3.6.1.4.1.14519.5.2.1.6279.6001.124822907934319930841506266464"]
    # # srslst = ["1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249",\
    # #           "1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094"]
    # #
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/data/preprocessed/subset9/"
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210126-162333/bbox/"
    #
    # ctdat = np.load(data_dir+srslst[showid]+'_clean.npy', allow_pickle=True)
    # ctlab = np.load(data_dir+srslst[showid]+'_label.npy', allow_pickle=True)

    ## results for methodist data
    # srslst = ["032873150-20131110",
    #           "015995871-20160929",
    #           "022467229-20180409",
    #           "028735272-20150702",
    #           "013746417-20130514"]

    # srslst = ["013484043-20180202",
    #           "033942251-20130925",
    #           "014776371-20171117"]

    # srslst = ["029337094-20120315",]
    # srslst = ["000192476-20160614",]
    # srslst = ["005203419-20160716",
    #           "102883931-20180417",
    #           "040779696-20160823",
    #           "007494206-20171016",
    #           "000192476-20160614",
    #           "001734722-20130821"]

    # srslst = ["patient002-20090310"]