"""
python data_utils.py get_id -s=Z:\Methodist_incidental_Kim\Dicom_data_additional_TC -c=Z:\Methodist_incidental_Kim -p=Z:\Methodist_incidental_Kim\Dicom_data -ext=Ben
python data_utils.py get_npz -r=Z:\Methodist_incidental_Kim\Dicom_data_additional_Ben -s=Z:\Methodist_incidental_Kim\Npz_data_additional_Ben -c=Z:\Methodist_incidental_Kim -n=False -ext=Ben

python data_utils.py get_id -s=Z:\Methodist_incidental_Kim\Dicom_data_additional -c=Z:\Methodist_incidental_Kim -p=Z:\additional_Data_TC -ext=TC
python data_utils.py move_data -s=Z:\Methodist_incidental_Kim\Dicom_data_additional_TC
python data_utils.py get_npz -r=Z:\Methodist_incidental_Kim\Dicom_data_additional_TC -s=Z:\Methodist_incidental_Kim\Npz_data_additional_TC -c=Z:\Methodist_incidental_Kim -p=Z:\processed_data\labeled_TC -n=True -ext=TC
python data_utils.py get_npz -r=Z:\Methodist_incidental_Kim\Dicom_data_additional_TC -s=Z:\Methodist_incidental_Kim\Npz_data_additional_TC -c=Z:\Methodist_incidental_Kim -n=False -ext=TC

python data_utils.py get_id -s=Z:\Methodist_incidental_Kim\Dicom_data_additional -c=Z:\Methodist_incidental_Kim -p=Z:\additional_Data_Frank -ext=Frank
python data_utils.py move_data -s=Z:\Methodist_incidental_Kim\Dicom_data_additional_Frank
python data_utils.py get_npz -r=Z:\Methodist_incidental_Kim\Dicom_data_additional_Frank -s=Z:\Methodist_incidental_Kim\Dicom_data_additional_Frank -c=Z:\Methodist_incidental_Kim -p=Z:\processed_data\labeled_Frank -n=True -ext=Frank
python data_utils.py get_npz -r=Z:\Methodist_incidental_Kim\Dicom_data_additional_Frank -s=Z:\Methodist_incidental_Kim\Dicom_data_additional_Frank -c=Z:\Methodist_incidental_Kim -n=False -ext=Frank

python data_utils.py get_id -s=Z:\Methodist_incidental_Mamta\Dicom_data -c=Z:\Methodist_incidental_Mamta -p=Z:\Data_mamta\Data_mamta -ext=Mamta
python data_utils.py get_npz -r=Z:\Methodist_incidental_Mamta\Dicom_data_Mamta -s=Z:\Methodist_incidental_Mamta\Npz_data_Mamta -c=Z:\Methodist_incidental_Mamta -n=True -ext=Mamta
python data_utils.py get_npz -r=Z:\Methodist_incidental_Mamta\Dicom_data_Mamta -s=Z:\Methodist_incidental_Mamta\Npz_data_Mamta -c=Z:\Methodist_incidental_Mamta -n=False -ext=Mamta

"""

from utils import read_slices, load_dicom, resample_image, Logger
from shutil import copyfile, copytree
from collections import Counter
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import shutil
import time
import sys
import os

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def find_date(annot):
    dates = []
    if not pd.isna(annot):
        start = list(find_all(annot, "{"))
        for s in start:
            assert annot[s + 9] == ":"
            dates.append(annot[s + 1: s + 9])
    return dates

def get_identifiers(save_dir, ck_dir, pre_dir=None, extra_str=""):
    """
    Get identifiers.csv which contains the identification information of the CT scans
    to be downloaded from the database

    :param pre_dir: Directory which saves the pre-downloaded data
    :param save_dir: Directory to save new data to be downloaded
    :param ck_dir: Directory of checklist file
    :param extra_str: Extra string to identify the checklist, e.g. "checklist_Ben" --> extra_str = "Ben"
    :return: None
    """
    if pre_dir is None: pre_dir = "./" 
    # date_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # basename = os.path.basename(save_dir)
    # basename = basename + "_{:s}".format(date_str)
    # if len(extra_str) > 0: basename = basename + "_{:s}".format(extra_str)
    # save_dir = os.path.join(os.path.dirname(save_dir), basename)
    if len(extra_str) > 0: extra_str = "_{:s}".format(extra_str)
    checklist_path = os.path.join(ck_dir, "checklist{:s}.xlsx".format(extra_str))
    # checklist_path = "Z:\Methodist_incidental_Kim\checklist_TC.xlsx"
    df = pd.read_excel(checklist_path, skiprows=1, dtype={"MRN": str, "date": str})
    df["MRN"] = df["MRN"].str.zfill(9)
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, "identifiers.csv"), "w")
    mf = open(os.path.join(save_dir, "move_data.csv"), "w")
    mfb = open(os.path.join(save_dir, "move_data_back.csv"), "w")
    identifiers = []
    all_identifiers = []
    for i in tqdm(range(len(df))):
        info = df.iloc[i]
        pstr = info["Patient\n Index"]
        mrn = info["MRN"]
        date = info["date"]
        id = "{:}-{:}-{:}".format(pstr, mrn, date)
        if id not in all_identifiers:
            # Process this scan
            all_identifiers.append(id)
            find_flag = False
            save_folder_name = "Lung_{:}-{:}_{:}".format(pstr, mrn, date)
            data_folder_name = "{:}_CT_data".format(date)
            save_path = os.path.join(save_dir, save_folder_name, data_folder_name)
            root_data_path = os.path.join(pre_dir, save_folder_name, data_folder_name)
            if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
                find_flag = True
            else:
                if os.path.exists(root_data_path) and len(os.listdir(root_data_path)) > 0:
                    find_flag = True
                    # copytree(root_data_path, save_path)
                    move_line = "{:}, {:}\n".format(root_data_path, save_path)
                    mf.write(move_line)
            if not find_flag:
                line = "{:},{:},{:}\n".format(pstr[7:], mrn, date)
                f.write(line)
                move_back_line = "{:}, {:}\n".format(save_path, root_data_path)
                mfb.write(move_back_line)
                identifiers.append(id)
    f.close()
    mf.close()
    mfb.close()
    print(np.array(identifiers))

def move_data(save_dir, reverse=False):
    if reverse:
        file = os.path.join(save_dir, "move_data_back.csv")
    else:
        file = os.path.join(save_dir, "move_data.csv")
    with open(file, "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            src, dst = l.split(",")
            try:
                copytree(src.strip(), dst.strip())
            except:
                copyfile(src.strip(), dst.strip())

def create_dataset_details(root_folder):
    data_folder = os.path.join(root_folder, "Data")
    xls_file = os.path.join(root_folder, "details.xlsx")
    df = pd.read_excel(xls_file, dtype={"MRN": str})
    df['MRN'] = df['MRN'].apply(lambda x: '{0:0>9}'.format(x))
    gt_df = pd.read_csv(os.path.join(root_folder, "gt_labels.csv"))

    from IncidentalData import LungDataset
    pos_label_file = "I:\Lung_ai\gt_labels.csv"
    cat_label_file = "I:\Lung_ai\Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    cube_size = 64
    lungData = LungDataset(data_folder, labeled_only=True, pos_label_file=pos_label_file,
                           cat_label_file=cat_label_file,
                           cube_size=cube_size, reload=False, screen=True)
    allpIDinLungDataset = [i["patientID"] for i in lungData.imageInfo]

    all_patients = [o for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, o))]
    all_patients = natsorted(all_patients)
    for i, patient in tqdm(enumerate(all_patients)):
        patient_folder = os.path.join(data_folder, patient)
        op_date = patient_folder.rsplit("-", 1)[1]
        df.iloc[i, -2] = op_date
        all_dates = [d[:8] for d in os.listdir(patient_folder) if
                     os.path.isdir(os.path.join(patient_folder, d)) and d[-4:] == "data"]
        all_dates = natsorted(all_dates)[::-1]
        if len(all_dates) > 0:
            df.iloc[i, -1] = "; ".join(all_dates)

        pstr = patient.split("-")[0].split("_")[1]
        # dstr = all_dates[j].split("_")[0]
        pID = patient.split("-")[1].split("_")[0]
        existId = (gt_df["patient"] == pstr)
        if existId.sum() == 0:
            continue

        assert df.iloc[i, 1] == pID
        Before_list, After_list = [], []
        dates_in_gt = gt_df[existId]["date"].to_numpy()
        counter = Counter(dates_in_gt)
        for key in counter.keys():
            if int(key) <= int(op_date):
                Before_list.append(str({key: counter[key]}))
            else:
                After_list.append(str({key: counter[key]}))

        df.iloc[i, 2] = 1
        if len(Before_list) > 0: df.iloc[i, 3] = ", ".join(Before_list)
        if len(After_list) > 0: df.iloc[i, 4] = ", ".join(After_list)
        if pID in allpIDinLungDataset: df.iloc[i, 5] = 1

    df.to_excel(os.path.join(root_folder, "Dataset_details_new.xlsx"), index=False)


    print("")

def add_scan(pstr, patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, pet, **kwargs):
    '''
    Add current scan meta information into global list
    :param: meta information for current scan
    :return: scan_info (in dictionary)
    '''
    scanInfo = {
        "pstr": pstr,
        "patientID": patientID,
        "scanID": scanID,
        "date": date,
        "series": series,
        "imagePath": imgPath,
        "sliceThickness": sliceThickness,
        "pixelSpacing": pixelSpacing,
        "PET": pet
    }
    scanInfo.update(kwargs)
    return scanInfo

def remove_empty_dir(dir):
    if len(os.listdir(dir)) == 0:
        shutil.rmtree(dir)



def load_from_dicom(root_dir, save_dir, ck_dir, pre_dir=None, normalize=True, extra_str=""):
    '''
    load image from dicom files
    :param root_dir: Directory contains the DICOM data
    :param save_dir: Directory to save npz data
    :param ck_dir: Directory contains the checklist file
    :param pre_dir: Directory contains the preprocessed npz data
    :param normalize: whether to normalize data
    :param extra_str: Extra string to identify the checklist, e.g. "checklist_Ben" --> extra_str = "Ben"
    :return: None
    '''
    root_dir = root_dir.rstrip("\\")
    # save_dir = os.path.join(root_dir.replace("Dicom", "Npz"))
    normalized_str = "normalized" if normalize else "raw"
    save_dir = os.path.join(save_dir, normalized_str)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log")
    sys.stdout = Logger(log_path)

    matches = ["LUNG", "lung"]
    no_CTscans = []
    matchMoreThanOne = []

    mf = open(os.path.join(save_dir, "move_data.csv"), "a+")
    sf = open(os.path.join(os.path.dirname(root_dir), "change_series.csv"), "a+")
    all_mf_lines = mf.readlines()
    all_sf_lines = sf.readlines()

    if len(extra_str) > 0: extra_str = "_{:s}".format(extra_str)
    checklist_df = pd.read_excel(os.path.join(ck_dir, "checklist{:s}.xlsx".format(extra_str)),
                                 skiprows=1, dtype={"date": str})
    if pre_dir is not None:
        addImageInfo = np.load(os.path.join(pre_dir, "CTinfo.npz"), allow_pickle=True)["info"].tolist()
        addScans = [d["imagePath"].replace("/", "\\").split("\\")[-1] for d in addImageInfo]
    try:
        imageInfo = np.load(os.path.join(root_dir, "CTinfo.npz"), allow_pickle=True)["info"].tolist()
    except:
        imageInfo = []
    all_patients = [i for i in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, i)) and i[:4] == "Lung"]
    all_patients = natsorted(all_patients)

    # Loop over all patients
    for i in tqdm(range(len(all_patients))):
        patientFolder = os.path.join(root_dir, all_patients[i])
        all_dates = [d for d in os.listdir(patientFolder) if
                     os.path.isdir(os.path.join(patientFolder, d)) and d[-4:] == "data"]
        # Loop over all dates
        for j in range(len(all_dates)):
            imgFolder = os.path.join(root_dir, all_patients[i], all_dates[j])
            pstr = all_patients[i].split("-")[0].split("_")[1]
            dstr = all_dates[j].split("_")[0]
            pID = all_patients[i].split("-")[1].split("_")[0]
            save_patient_dir = os.path.join(save_dir, all_patients[i].rsplit('_', 1)[0])
            os.makedirs(save_patient_dir, exist_ok=True)
            image_path = os.path.join(save_patient_dir, "{:s}-{:s}.npz".format(pID, dstr))

            if os.path.exists(image_path):
                continue

            id = image_path.split("\\")[-1]
            if pre_dir is not None:
                if id in addScans:
                    idx = addScans.index(id)
                    imagePath = addImageInfo[idx]["imagePath"]
                    assert os.path.exists(imagePath), "{:} does not exist!".format(imagePath)
                    move_line = "{:}, {:}\n".format(imagePath, image_path)
                    if move_line not in all_mf_lines: mf.write(move_line)
                    remove_empty_dir(save_patient_dir)
                    continue

            # find series
            existId = (checklist_df["Patient\n Index"] == pstr) & (checklist_df["date"] == dstr)
            series = checklist_df[existId]["Series"].to_numpy()
            if len(series) > 0:
                series = series[0]
                if series == "Lung_Bone+ 50cm" or series == "LUNG_BONE PLUS 50cm":
                    series = series.replace("_", "/")
                pet = checklist_df[existId]["PET"].to_numpy()[0]
            else:
                # series not found in checklist
                continue

            # Distribute all slices to different series
            patientID, dateDicom, seriesDict = load_dicom(imgFolder)
            if patientID != pID:
                print("PatientID does not match!! patientID: {:s} -- pID: {:s}".format(patientID, pID))
            if dateDicom != dstr:
                print("Date does not match!!  dateDicom: {:s} -- dstr: {:s}".format(dateDicom, dstr))
            print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))
            print("All series types: ", list(seriesDict.keys()))

            if series not in seriesDict:
                lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in matches])]
                if len(lungSeries) == 0:
                    print("No lung scans found!")
                    no_CTscans.append(seriesDict)
                    remove_empty_dir(save_patient_dir)
                    continue
                else:
                    if len(lungSeries) > 1:
                        print("More than 1 lung scans found!")
                        id = np.argmin([len(i) for i in lungSeries])
                        series = lungSeries[id]
                        matchMoreThanOne.append(lungSeries)
                    else:
                        series = lungSeries[0]
                    info = checklist_df[existId].iloc[0]
                    change_line = "{:}, {:}, {:}\n".format(info["Patient\n Index"], info["Series"], series)
                    if change_line not in all_sf_lines: sf.write(change_line)
            print("Lung series: ", series)

            # Load and save lung series
            slices = seriesDict[series]
            image, sliceThickness, pixelSpacing, scanID = read_slices(slices)

            scanInfo = add_scan(pstr, patientID, dateDicom, series, image_path,
                                sliceThickness, pixelSpacing, scanID, pet)
            imageInfo.append(scanInfo)
            if normalize:
                new_image, new_spacing = resample_image(image, sliceThickness, pixelSpacing)
                np.savez_compressed(image_path, image=new_image, info=scanInfo)
            else:
                np.savez_compressed(image_path, image=image, info=scanInfo)
            print("Save scan to {:s}".format(image_path))

            print("\nFinish loading patient {:s} at date {:s} <<<<<<<".format(patientID, dateDicom))

            CTinfoPath = os.path.join(save_dir, "CTinfo.npz")
            np.savez_compressed(CTinfoPath, info=imageInfo)
            print("Save all scan infos to {:s}".format(CTinfoPath))

    mf.close()
    sf.close()

    print("-" * 30 + " CTinfo " + "-" * 30)
    [print(i) for i in imageInfo]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data_prepare")
    parser.add_argument("command", type=str, help="options: get_id, move_data, get_npz")
    parser.add_argument("-r", "--root_dir", type=str, default="./", help="Root directory")
    parser.add_argument("-s", "--save_dir", type=str, help="Save directory")
    parser.add_argument("-c", "--ck_dir", type=str, help="Checklist directory")
    parser.add_argument("-p", "--pre_dir", type=str, help="Directory where some data is preprocessed/pre-downloaded")
    parser.add_argument("-n", "--normalize", type=eval, default=True, help="Whether to normalize data")
    parser.add_argument("-rev", "--reverse", type=eval, default=False, help="If True, move data from save_dir to pre_dir")
    parser.add_argument("-ext", "--extra_str", type=str, default="", help="Extra string for the data")
    args = parser.parse_args()

    if args.command == "get_id":
        get_identifiers(save_dir=args.save_dir, ck_dir=args.ck_dir, pre_dir=args.pre_dir, extra_str=args.extra_str)
    elif args.command == "move_data":
        move_data(save_dir=args.save_dir, reverse=args.reverse)
    elif args.command == "get_npz":
        load_from_dicom(root_dir=args.root_dir, save_dir=args.save_dir, ck_dir=args.ck_dir, pre_dir=args.pre_dir,
                        normalize=args.normalize, extra_str=args.extra_str)