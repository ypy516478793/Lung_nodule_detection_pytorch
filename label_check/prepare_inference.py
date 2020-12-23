import numpy as np
import os

# root_dir = "/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/label_check/model_predictions_Kim"
# for dirName, subdirList, fileList in os.walk(root_dir):
#     print("Found directory: %s" % dirName)
#     for fname in fileList:
#         print("\t%s" % fname)
#
#     # if os.path.basename(dirName) == "checkpoint":
#     #     epoch_list = np.array([int(fname.strip("model-").split(".")[0]) for fname in fileList if fname[:6] == "model-"])
#     #     file_list = np.array([fname for fname in fileList if fname[:6] == "model-"])
#     #     if len(epoch_list) > 0:
#     #         mask = epoch_list != np.max(epoch_list)
#     #         remove_ids = np.arange(len(mask))[mask]
#     #         for i in remove_ids:
#     #             cp_path = os.path.join(dirName, file_list[i])
#     #             print("delete {:s}".format(cp_path))
#     #             os.remove(cp_path)
#     h5_list = [fname for fname in fileList if fname[:3] == "pbb" and fname.endswith(".npy")]
#     for i in h5_list:
#         cp_path = os.path.join(dirName, i)
#         print("delete {:s}".format(cp_path))
#         os.remove(cp_path)

from ast import literal_eval
import pandas as pd
root_dir = "/Users/yuan_pengyu/PycharmProjects/DeepLung-3D_Lung_Nodule_Detection/label_check/model_predictions_Kim"

xls_file = os.path.join(root_dir, "nodule_details.xlsx")
# df = pd.read_excel(xls_file, dtype={"MRN": str})
# df["MRN"] = df["MRN"].apply(lambda x: "{0:0>9}".format(x))
# gt_df = pd.read_csv(os.path.join(root_dir, "gt_labels.csv"))

nodule_infos = []
folders = [i for i in os.listdir(root_dir) if "-" in i]
for folder in folders:
    mrn, date = folder.split("-")
    result_txt = os.path.join(root_dir, folder, "pbb_ori.txt")
    with open(result_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Patient"):
                mrn2, date2 = line.split(" ")[-1].strip("\n").split("-")
                assert mrn == mrn2
                assert date == date2
            nodule_idx = line.split(":")[0]
            if nodule_idx.isdigit():
                temp = line.split(":")[-1].split("[")[1].split("]")[0]
                label = [float(i.strip(".")) for i in temp.split()]
                # label = literal_eval(line.split(":")[-1])
                info = [mrn, date, nodule_idx, np.nan, *label]
                nodule_infos.append(info)

df = pd.DataFrame(nodule_infos, columns = ["MRN", "Date", "Index", "Nodule", "z", "y", "x", "d"])
df.to_excel(os.path.join("./", "nodule_info.xlsx"), index=False)

print("")




#
# from IncidentalData import LungDataset
# pos_label_file = "I:\Lung_ai\gt_labels.csv"
# cat_label_file = "I:\Lung_ai\Lung Nodule Clinical Data_Min Kim (No name).xlsx"
# cube_size = 64
# lungData = LungDataset(data_folder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
#                        cube_size=cube_size, reload=False, screen=True)
# allpIDinLungDataset = [i["patientID"] for i in lungData.imageInfo]
#
# all_patients = [o for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,o))]
# all_patients = natsorted(all_patients)
# for i, patient in tqdm(enumerate(all_patients)):
#     patient_folder = os.path.join(data_folder, patient)
#     op_date = patient_folder.rsplit("-", 1)[1]
#     df.iloc[i, -2] = op_date
#     all_dates = [d[:8] for d in os.listdir(patient_folder) if
#                  os.path.isdir(os.path.join(patient_folder, d)) and d[-4:] == "data"]
#     all_dates = natsorted(all_dates)[::-1]
#     if len(all_dates) > 0:
#         df.iloc[i, -1] = "; ".join(all_dates)
#
#
#     pstr = patient.split("-")[0].split("_")[1]
#     # dstr = all_dates[j].split("_")[0]
#     pID = patient.split("-")[1].split("_")[0]
#     existId = (gt_df["patient"] == pstr)
#     if existId.sum() == 0:
#         continue
#
#     assert df.iloc[i, 1] == pID
#     Before_list, After_list = [], []
#     dates_in_gt = gt_df[existId]["date"].to_numpy()
#     counter = Counter(dates_in_gt)
#     for key in counter.keys():
#         if int(key) <= int(op_date):
#             Before_list.append(str({key: counter[key]}))
#         else:
#             After_list.append(str({key: counter[key]}))
#
#     df.iloc[i, 2] = 1
#     if len(Before_list) > 0: df.iloc[i, 3] = ", ".join(Before_list)
#     if len(After_list) > 0: df.iloc[i, 4] = ", ".join(After_list)
#     if pID in allpIDinLungDataset: df.iloc[i, 5] = 1
#
#
# df.to_excel(os.path.join(root_dir, "Dataset_details_new.xlsx"), index=False)