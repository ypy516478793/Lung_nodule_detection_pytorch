import pandas as pd
details ="/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/dataset_details_v0.xlsx"
xls = pd.ExcelFile(details)
df0 = pd.read_excel(xls, 'Patient-level')
df1 = pd.read_excel(xls, 'Scan-level')
df2 = pd.read_excel(xls, 'Nodule-level')
df2_label = df2[df2["d"] != None]
df2_label = df2[~df2["d"].isna()]
pstrs = df2_label["Patient Index"].values.tolist()
dstrs = df2_label["date"].values.tolist()
ids = [i+"_"+str(j) for i, j in zip(pstrs, dstrs)]

pstrs = df0["Patient index"].values.tolist()
mags = df0["Category Of Disease - Primary {1300} (1=lung cancer, 2=metastatic, 3 = benign nodule, 4= bronchiectasis/pulm sequestration/infection)"].values.tolist()
mag_dict = {}
for i, j in zip(pstrs, mags):
    if "/" in i:
        i0 = i.split("/")[0]
        i1 = "patient" + i.split("/")[1]
        mag_dict[i0] = j
        mag_dict[i1] = j
    else:
        mag_dict[i] = j

from collections import Counter
cnt = Counter(ids)
for i in range(len(df1)):
    id = df1.iloc[i]["Patient Index"] + "_" + str(df1.iloc[i]["date"])
    pstr = df1.iloc[i]["Patient Index"]
    if id in cnt:
        df1.loc[i, "Number of nodules"] = int(cnt[id])
    df1.loc[i, "Patient malignancy"] = mag_dict[pstr]

df1.to_excel("tmp.xlsx")