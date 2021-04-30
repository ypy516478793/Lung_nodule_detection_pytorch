import matplotlib
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import os
showid = 1 # from 0 to 4
assert showid in range(5)

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype("uint8")
    return newimg

def resample_pos(label, thickness, spacing, new_spacing=[1, 1, 1]):
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    resize_factor = resize_factor[::-1]
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]
    return label

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
        ax.imshow(images[int(zl)], cmap="gray")
        rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                       facecolor='none')
        ax.add_patch(rect_label)
        if np.abs(zp - zl) <= dp:
            rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                          facecolor='none')
            ax.add_patch(rect_pred)
    elif label is not None:
        zl, yl, xl, dl = label
        ax.imshow(images[int(zl)], cmap="gray")
        rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                       facecolor='none')
        ax.add_patch(rect_label)
    elif pred is not None:
        zp, yp, xp, dp = pred
        ax.imshow(images[int(zp)], cmap="gray")
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
srslst = ["005203419-20160716",
          "102883931-20180417",
          "040779696-20160823",
          "007494206-20171016",
          "000192476-20160614",
          "001734722-20130821"]

# data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
# data_dir = "/data/pyuan2/Methodist_incidental/data_kim/labeled/"
# data_dir = "/data/pyuan2/Methodist_incidental/data_kim/masked_first/"
# data_dir = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/masked_with_crop/"
data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug/"
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
result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210429-200058-train/bbox/"
show = True

pos_label_file = "pos_labels_norm.csv"
info_file = "CTinfo.npz"
pos_df = pd.read_csv(os.path.join(data_dir, pos_label_file), dtype={"date": str})
imageInfo = np.load(os.path.join(data_dir, info_file), allow_pickle=True)["info"]

patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
                 for info, id in zip(imageInfo, np.arange(len(imageInfo)))}
imageId = patient2Image[srslst[showid]]
filename = imageInfo[imageId]["imagePath"]
pstr = imageInfo[imageId]["pstr"]
dstr = imageInfo[imageId]["date"]
thickness = imageInfo[imageId]["sliceThickness"]
spacing = imageInfo[imageId]["pixelSpacing"]
existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
pos = pos_df[existId]
temp = pos[["x", "y", "z", "d"]].values
# temp[:, 2] = temp[:, 2] - 1
# temp = np.array([resample_pos(p, thickness, spacing) for p in temp])
ctlab = temp[:, [2, 1, 0, 3]]
# ctlab[:, 0] = ctlab[:, 0] - 1

try:
    imgs = np.load(filename, allow_pickle=True)["image"][np.newaxis, :]
except Exception:
    imgs = np.load(filename.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
    extendbox = np.load(filename.replace(".npz", "_extendbox.npz"))["extendbox"]

    if len(ctlab) == 0:
        ctlab = np.array([[0, 0, 0, 0]])
    else:
        ll = np.copy(ctlab).T
        # label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        # label2[3] = label2[3] * spacing[1] / resolution[1]
        ll[:3] = ll[:3] - np.expand_dims(extendbox[:, 0], 1)
        ctlab = ll[:4].T
# ctdat = lumTrans(imgs)
ctdat = imgs

print('Groundtruth')
print("image shape is: ", ctdat.shape)
print("label shape is: ", ctlab.shape)
# print(ctdat.shape, ctlab.shape)

# plt.rcParams.update({'font.size': 5})

for idx in range(ctlab.shape[0]):
    if abs(ctlab[idx,0])+abs(ctlab[idx,1])+abs(ctlab[idx,2])+abs(ctlab[idx,3])==0: continue
    title = "series {:s} \n label {:s}".format(srslst[showid], str(ctlab[idx]))
    plot_bbox(title, ctdat[0], None, ctlab[idx], show=show, title=title, fontsize=10 )
    # fig = plt.figure()
    # z, x, y = int(ctlab[idx,0]), int(ctlab[idx,1]), int(ctlab[idx,2])
    # dat0 = np.array(ctdat[0, z, :, :])
    # dat0[max(0,x-10):min(dat0.shape[0],x+10), max(0,y-10)] = 255
    # dat0[max(0,x-10):min(dat0.shape[0],x+10), min(dat0.shape[1],y+10)] = 255
    # dat0[max(0,x-10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
    # dat0[min(dat0.shape[0],x+10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
    # plt.imshow(dat0, cmap="gray")
    # plt.title("series {:s} \n label {:s}".format(srslst[showid], str(ctlab[idx])), fontsize=10)
plt.show()


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
pbb = np.load(result_dir+srslst[showid]+'_pbb.npy')
lbb = np.load(result_dir+srslst[showid]+'_lbb.npy')
# pbb = np.load(result_dir+srslst[showid]+'_image_pbb.npy')
# lbb = np.load(result_dir+srslst[showid]+'_image_lbb.npy')

pbb = np.array(pbb[pbb[:,0] > 0])
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
    title = "series {:s} \n predict {:s}".format(srslst[showid], str(pbb[idx]))
    plot_bbox(title, ctdat[0], pbb[idx][1:], None, title=title, show=show, fontsize=10 )
#     fig = plt.figure()
#     z, x, y = int(pbb[idx,1]), int(pbb[idx,2]), int(pbb[idx,3])
# #     print z,x,y
#     dat0 = np.array(ctdat[0, z, :, :])
#     dat0[max(0,x-10):min(dat0.shape[0]-1,x+10), max(0,y-10)] = 255
#     dat0[max(0,x-10):min(dat0.shape[0]-1,x+10), min(dat0.shape[1]-1,y+10)] = 255
#     dat0[max(0,x-10), max(0,y-10):min(dat0.shape[1]-1,y+10)] = 255
#     dat0[min(dat0.shape[0]-1,x+10), max(0,y-10):min(dat0.shape[1]-1,y+10)] = 255
#     plt.imshow(dat0, cmap="gray")
#     plt.title("series {:s} \n predict {:s}".format(srslst[showid], str(pbb[idx])), fontsize=10)
plt.show()

print("")
