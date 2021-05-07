import tensorflow as tf
import numpy as np

def compute_ap_range(gt_box,
                     pred_box, pred_score,
                     iou_thresholds=None, score_threshold=0.0, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box,
                       pred_box, pred_score,
                       iou_threshold=iou_threshold,
                       score_threshold=score_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_ap(gt_boxes,
               pred_boxes,  pred_scores,
               iou_threshold=0.5, score_threshold=0.0):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes,
        pred_boxes, pred_scores,
        iou_threshold, score_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

def compute_matches(gt_boxes,
                    pred_boxes, pred_scores,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_bboxes(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break

    return gt_match, pred_match, overlaps


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def compute_overlaps_bboxes(bboxes1, bboxes2):
    """
    bboxes1, bboxes2: [instances, (z, y, x, d)]
    """
    num_preds = len(bboxes1)
    num_gts = len(bboxes2)
    overlaps = np.zeros([num_preds, num_gts])
    for i in range(num_preds):
        for j in range(num_gts):
            overlaps[i, j] = iou(bboxes1[i], bboxes2[j])

    return overlaps


def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


# def compute_overlaps_masks(masks1, masks2):
#     """Computes IoU overlaps between two sets of masks.
#     masks1, masks2: [Height, Width, instances]
#     """
#
#     # If either set of masks is empty return empty result
#     if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
#         return np.zeros((masks1.shape[-1], masks2.shape[-1]))
#     # flatten masks and compute their areas
#     masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
#     masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
#     area1 = np.sum(masks1, axis=0)
#     area2 = np.sum(masks2, axis=0)
#
#     # intersections and union
#     intersections = np.dot(masks1.T, masks2)
#     union = area1[:, None] + area2[None, :] - intersections
#     overlaps = intersections / union
#
#     return overlaps


# find index of pbb:
# np.where(np.sum(np.abs(pbb[:, 1:] - np.array([147.81473,  161.29596,  183.44478,  25.386171])), axis=-1) < 0.1)


lfile = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210119-113503/bbox/001030196-20121205.npz_lbb.npy"
pfile = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210119-113503/bbox/001030196-20121205.npz_pbb.npy"
pbb = np.load(pfile)

conf_th = 4
nms_th = 0.5
pbb = pbb[pbb[:, 0] >= conf_th]
pbb = nms(pbb, nms_th)



gt_bboxes = np.load(lfile)
pred_bboxes = pbb[:, 1:]
pred_scores = pbb[:, 0]


ap = compute_ap_range(gt_bboxes,
                    pred_bboxes, pred_scores,
                       verbose=1)





print("")