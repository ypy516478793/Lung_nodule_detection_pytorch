import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

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

def add_bbox(ax, images, pred, label=None):
    if len(images.shape) == 2:
        im = ax.imshow(images, cmap="gray")
        if label is not None and pred is not None:
            yp, xp, dp = pred
            yl, xl, dl = label
            rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                           facecolor='none')
            ax.add_patch(rect_label)
            rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                          facecolor='none')
            ax.add_patch(rect_pred)
        elif label is not None:
            yl, xl, dl = label
            rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                           facecolor='none')
            ax.add_patch(rect_label)
        elif pred is not None:
            yp, xp, dp = pred
            rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                          facecolor='none')
            ax.add_patch(rect_pred)
        else:
            print("no prediction or label is given!")
    else:
        assert len(images.shape) == 3
        if label is not None and pred is not None:
            zp, yp, xp, dp = pred
            zl, yl, xl, dl = label
            im = ax.imshow(images[int(zl)], cmap="gray")
            rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                           facecolor='none')
            ax.add_patch(rect_label)
            if np.abs(zp - zl) <= dp:
                rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                              facecolor='none')
                ax.add_patch(rect_pred)
        elif label is not None:
            zl, yl, xl, dl = label
            im = ax.imshow(images[int(zl)], cmap="gray")
            rect_label = patches.Rectangle((xl - dl / 2, yl - dl / 2), dl, dl, linewidth=1, edgecolor='g',
                                           facecolor='none')
            ax.add_patch(rect_label)
        elif pred is not None:
            zp, yp, xp, dp = pred
            im = ax.imshow(images[int(zp)], cmap="gray")
            rect_pred = patches.Rectangle((xp - dp / 2, yp - dp / 2), dp, dp, linewidth=1, edgecolor='r',
                                          facecolor='none')
            ax.add_patch(rect_pred)
        else:
            print("no prediction or label is given!")
    return im

def plot_bbox(savedir, images, pred, label=None, show=True, title=None):
    '''
    plot center image with bbox
    :param images: CT scan, shape: (num_slices, h, w) or (h, w)
    :param label: coordinates & diameter (all in pixel space): (z, y, x, d) or (y, x, d)
    :param savedir: save directory
    :return: None
    '''
    fig, ax = plt.subplots(1)
    im = add_bbox(ax, images, pred, label)
    fig.colorbar(im)
    if title:
        plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(savedir + "_bbox.png")
        plt.close()

def plot_luna_raw(data_path):
    '''
    data_path: endswith .mhd
    '''
    from prepare import load_itk_image, resample
    sliceim, origin, spacing, isflip = load_itk_image(data_path)
    resolution = np.array([1, 1, 1])
    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
        print('flip!')
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)


def draw_bbox(savedir, images, pred, label=None):
    '''
    plot center image with bbox
    :param images: CT scan, shape: (num_slices, h, w) or (h, w)
    :param label: coordinates & diameter (all in pixel space): (x, y, z, d) or (x, y, d)
    :param savedir: save directory
    :return: None
    '''
    fig, ax = plt.subplots(1)
    im = add_bbox(ax, images, pred, label)
    from detector.utils import canvas2array
    fig.canvas.draw()
    data = canvas2array(fig)
    plt.close(fig)
    plt.imshow(data)
    plt.show()
    return data

def plot_learning_curve():
    train_loss = [0.0305, 0.2446, 0.3404, 0.1859, 0.2307, 0.1591, 0.1614, 0.2249, 0.2300, 0.1515]
    test_loss = [0.3969, 0.3477, 0.4032, 0.2483, 0.2044, 0.1972, 0.2259, 0.2266, 0.2338, 0.1973]
    train_acc = [100, 94.23, 83.33, 90.38, 90.70, 92.86, 93.72, 95.30, 92.74, 93.24]
    test_acc = [89.69, 91.93, 91.03, 93.72, 93.27, 93.27, 92.38, 93.72, 93.72, 94.17]

    train_size = np.arange(10, 105, 10)
    plt.plot(train_size, train_loss, label="train_loss")
    plt.plot(train_size, test_loss, label="test_loss")
    # plt.plot(train_size, train_acc, label="train_acc")
    # plt.plot(train_size, test_acc, label="test_acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # showid = 0
    # srslst = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.208737629504245244513001631764',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.108231420525711026834210228428',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.161002239822118346732951898613',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.162901839201654862079549658100',\
    #           '1.3.6.1.4.1.14519.5.2.1.6279.6001.179162671133894061547290922949']
    # ctdat = np.load('./CT/'+srslst[showid]+'_clean.npy')
    # ctlab = np.load('./CT/'+srslst[showid]+'_label.npy')

    # pbb = np.load('./detection/'+srslst[showid]+'_pbb.npy')
    # lbb = np.load('./detection/'+srslst[showid]+'_lbb.npy')
    # 031787708
    ctdat = np.load("/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/Lung_patient436-013696729_20180326-20180626/013696729-20180420.npz")
    pbb = np.load("/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector/results/res18-20201117-134257/bbox/013696729-20180420.npz_pbb.npy")
    lbb = np.load("/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector/results/res18-20201117-134257/bbox/013696729-20180420.npz_lbb.npy")

    ctdat = ctdat["image"]
    pbb = np.array(pbb[pbb[:,0] > -2])
    pbb = nms(pbb, 0.1)
    # print pbb.shape, pbb
    plot_bbox(None, ctdat, pbb[0, 1:])
    print('Detection Results according to confidence')
    for idx in range(pbb.shape[0]):
        fig = plt.figure()
        # z, x, y = int(pbb[idx,1]), int(pbb[idx,2]), int(pbb[idx,3])

        z, x, y, d = lbb[0].astype(np.int)
    #     print z,x,y
        dat0 = np.array(ctdat[z, :, :])
        dat0[max(0,x-10):min(dat0.shape[0],x+10), max(0,y-10)] = 255
        dat0[max(0,x-10):min(dat0.shape[0],x+10), min(dat0.shape[1],y+10)] = 255
        dat0[max(0,x-10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
        dat0[min(dat0.shape[0],x+10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
        plt.imshow(dat0)

    print("")

    # plot_learning_curve()