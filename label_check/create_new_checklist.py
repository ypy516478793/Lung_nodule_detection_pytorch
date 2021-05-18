from tqdm import tqdm
import pandas as pd
import numpy as np
import os

def create_checklist_model():
    columns = ['Patient\n Index', 'MRN', 'date',
           'x', 'y', 'z', 'd', 'probability', 'nodule\nIndex']


    # checklist_path = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/checklist_unlabeled_small.xlsx"
    # # checklist_path = os.path.join(ck_dir, "checklist{:s}.xlsx".format(extra_str))
    # # checklist_path = "Z:\Methodist_incidental_Kim\checklist_TC.xlsx"
    # checklist_df = pd.read_excel(checklist_path, skiprows=1, dtype={"MRN": str, "date": str})
    # checklist_df["MRN"] = checklist_df["MRN"].str.zfill(9)



    data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
    result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled_0514/preds"
    namelist = np.load(os.path.join(result_dir, "namelist.npy"))
    print("")

    imageInfo = np.load(os.path.join(data_dir, "CTinfo.npz"), allow_pickle=True)["info"]

    patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
                     for info, id in zip(imageInfo, np.arange(len(imageInfo)))}

    # existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
    # pos = pos_df[existId]
    # temp = pos[["x", "y", "z", "d"]].values

    data = []
    not_include = []
    for name in tqdm(namelist):
        if name not in patient2Image:
            not_include.append(name)
        else:
            imageId = patient2Image[name]

        pred_dir = os.path.join(result_dir, name)
        pbb = np.load(os.path.join(pred_dir, "pbb.npy"))

        mrn = name.split("-")[0]
        filename = imageInfo[imageId]["imagePath"]
        pstr = imageInfo[imageId]["pstr"]
        dstr = imageInfo[imageId]["date"]
        thickness = imageInfo[imageId]["sliceThickness"]
        spacing = imageInfo[imageId]["pixelSpacing"]

        # extendbox = np.load(filename.replace(".npz", "_extendbox.npz"))["extendbox"]
        for i in range(len(pbb)):
            pbb[i, 0] = 1 / (1 + np.exp(-pbb[i, 0]))
            p, z, y, x, d = pbb[i]
            row = [pstr, mrn, dstr, x, y, z, d, p, i]
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    save_path = os.path.join(result_dir, "checklist_model.xlsx")
    df.to_excel(save_path, index=False)
    print("Saved to {:s}".format(save_path))

    print("Not included in data_dir: ", not_include)

if __name__ == '__main__':
    create_checklist_model()

# def convert_checklist():
#     checklist_model = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled_0514/preds/checklist_model.xlsx"
#     checklist_gt = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/checklist_unlabeled_small.xlsx"
#
#     data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
#     result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/Kim_unlabeled_0514/preds"
#     namelist = np.load(os.path.join(result_dir, "namelist.npy"))
#     print("")
#
#     imageInfo = np.load(os.path.join(data_dir, "CTinfo.npz"), allow_pickle=True)["info"]
#
#     patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
#                      for info, id in zip(imageInfo, np.arange(len(imageInfo)))}
#
#     checklist_df = pd.read_excel(checklist_real, skiprows=1, dtype={"MRN": str, "date": str})
#     checklist_df["MRN"] = checklist_df["MRN"].str.zfill(9)
#     for index, row in df.iterrows():
#
#
#     data = []
#     not_include = []
#     for name in tqdm(namelist):
#         if name not in patient2Image:
#             not_include.append(name)
#         else:
#             imageId = patient2Image[name]
#
#         pred_dir = os.path.join(result_dir, name)
#         pbb = np.load(os.path.join(pred_dir, "pbb.npy"))
#
#         mrn = name.split("-")[0]
#         filename = imageInfo[imageId]["imagePath"]
#         pstr = imageInfo[imageId]["pstr"]
#         dstr = imageInfo[imageId]["date"]
#         thickness = imageInfo[imageId]["sliceThickness"]
#         spacing = imageInfo[imageId]["pixelSpacing"]
#
#         # extendbox = np.load(filename.replace(".npz", "_extendbox.npz"))["extendbox"]
#         for i in range(len(pbb)):
#             pbb[i, 0] = 1 / (1 + np.exp(-pbb[i, 0]))
#             p, z, y, x, d = pbb[i]
#             row = [pstr, mrn, dstr, x, y, z, d, p, i]
#             data.append(row)
