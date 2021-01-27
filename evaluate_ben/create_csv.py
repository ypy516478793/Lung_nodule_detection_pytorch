from evaluationScript.tools import csvTools
import numpy as np
import csv


file = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210126-170116/bbox/namelist.npy"
a = np.load(file)


# annotations = csvTools.readCSV("../evaluationScript/annotations/annotations.csv")
# annotations_subset9 = []
# annotations_subset9.append(annotations[0])
# for anno in annotations[1:]:
#     if anno[0] in a:
#         annotations_subset9.append(anno)
# print(len(annotations_subset9))
# wfile = open("../evaluationScript/annotations/annotations_subset9.csv", "w")
# writer = csv.writer(wfile)
# for line in annotations_subset9:
#     writer.writerow(line)
# wfile.close()


# exclude_annotations = csvTools.readCSV("../evaluationScript/annotations/annotations_excluded.csv")
# annotations_exclude_subset9 = []
# annotations_exclude_subset9.append(exclude_annotations[0])
# for anno in exclude_annotations[1:]:
#     if anno[0] in a:
#         annotations_exclude_subset9.append(anno)
# print(len(annotations_exclude_subset9))
# wfile = open("../evaluationScript/annotations/annotations_exclude_subset9.csv", "w")
# writer = csv.writer(wfile)
# for line in annotations_exclude_subset9:
#     writer.writerow(line)
# wfile.close()



wfile = open("../evaluationScript/annotations/seriesuids_subset9.csv", "w")
writer = csv.writer(wfile)
for line in a:
    writer.writerow([line])
wfile.close()