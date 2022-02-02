from utils.data_utils import load_nii
import pandas as pd
import numpy as np
import glob, os

def convert_nifti_label():
    """
    Convert Nifti label: aggregate .nii -> .csv
    Args:
        label_dir: ./Methodist_incidental/data_Kelvin/Nifti_data_label
        save_dir: ./Methodist_incidental/data_Kelvin/Nifti_data
        [label_path: ./Methodist_incidental/data_Kelvin/Nifti_data_label/Lung_patient002/patient002_20090310.nii (e.g.)]
    Return:
        None
    """

    label_ls = glob.glob(label_dir + "/*/*.nii")
    label_ls.sort()
    labels = []
    for label_path in label_ls:

        fname = os.path.basename(label_path).rstrip(".nii")
        pstr, dstr = fname.split("_")
        Series = "Unknown"

        label, _ = load_nii(label_path)

        zs, ys, xs = np.where(label != 0)
        bbox_candidates = []
        for i in range(len(zs)):
            p = np.array([zs[i], ys[i], xs[i]])
            match = False
            for bc in bbox_candidates:
                l, r = bc[0], bc[1]
                ld = np.sum(np.abs(l - p))
                rd = np.sum(np.abs(r - p))
                if not np.all(l == r) and ld == 1 and rd == 1:
                    bc[3] = p  # end point
                    assert bc[2][0] == bc[3][0]  # make sure the slice index is the same
                    z = bc[2][0]
                    y = (bc[2][1] + bc[3][1]) / 2
                    x = (bc[2][2] + bc[3][2]) / 2
                    d = np.max([np.abs(bc[3][1] - bc[2][1]), np.abs(bc[3][2] - bc[2][2])])
                    labels.append((pstr, dstr, Series, x, y, z, d))
                    bbox_candidates.remove(bc)
                    match = True
                    break
                elif ld == 1:
                    bc[0] = p  # left point
                    match = True
                    break
                elif rd == 1:
                    bc[1] = p  # right point
                    match = True
                    break
            if not match:
                bc = [np.array(p)] * 4  # initial a bbox candidate
                bbox_candidates.append(bc)

    columns = ["patient", "date", "series", "x", "y", "z", "d"]
    label_df = pd.DataFrame(labels, columns=columns)
    label_df.to_csv(os.path.join(save_dir, "pos_labels_raw_v2.csv"), index=False)

if __name__ == '__main__':
    label_dir = "./Methodist_incidental/data_Kelvin/Nifti_data_label_v2"
    save_dir = "./Methodist_incidental/data_Kelvin/Nifti_data"
    convert_nifti_label()