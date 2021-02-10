from matplotlib.ticker import FixedFormatter
from NoduleFinding import NoduleFinding
from tqdm import tqdm
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'


FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve
bLogPlot = True


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

def convertcsv(bboxfname, result_dir, detp):
    # sliceim,origin,spacing,isflip = load_itk_image(datapath+bboxfname[:-8]+'.mhd')
    # origin = np.load(sideinfopath+bboxfname[:-8]+'_origin.npy', mmap_mode='r')
    # spacing = np.load(sideinfopath+bboxfname[:-8]+'_spacing.npy', mmap_mode='r')
    # resolution = np.array([1, 1, 1])
    # extendbox = np.load(sideinfopath+bboxfname[:-8]+'_extendbox.npy', mmap_mode='r')
    pbb = np.load(result_dir+bboxfname, mmap_mode='r')

    pbbold = np.array(pbb[pbb[:,0] > detp])
    pbbold = np.array(pbbold[pbbold[:,-1] > 3])  # add new 9 15
    # print(pbbold.shape)
    # pbb = np.array(pbb[:K, :4])
    # print pbbold.shape1
    # if use_softnms:
    #     keep = cpu_soft_nms(pbbold, method=2) # 1 for linear weighting, 2 for gaussian weighting
    #     pbb = np.array(pbbold[keep]) #cpu_soft_nms(pbbold)
    # else:
    pbb = nms(pbbold, nmsthresh)
    # print len(pbb), pbb[0]
    # print bboxfname, pbbold.shape, pbb.shape, pbbold.shape
    # pbb = np.array(pbb[:, :-1])
    # print pbb[:, 0]
    # pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T)
    # pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    # if isflip:
    #     Mask = np.load(sideinfopath+bboxfname[:-8]+'_mask.npy', mmap_mode='r')
    #     pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
    #     pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
    # pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    # print pos.shape
    for nk in range(pbb.shape[0]): # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([bboxfname[:-8], pbb[nk, 3], pbb[nk, 2], pbb[nk, 1], pbb[nk, 4], 1/(1+np.exp(-pbb[nk,0]))])
    # print len(rowlist), len(rowlist[0])
    return rowlist#bboxfname[:-8], pos[:K, 2], pos[:K, 1], pos[:K, 0], 1/(1+np.exp(-pbb[:K,0]))

def getcsv(detp, nmsthresh, ostr):
    firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability']
    bbox_dir = result_dir + "bbox/"
    for nmsth in nmsthresh:
        for detpthresh in detp:
            print('detp', detpthresh, "nmsth", nmsth)
            f = open(bbox_dir + ostr+ "_detp{:s}_nms{:s}".format(str(detpthresh), str(nmsth)) + '.csv', 'w')
            fwriter = csv.writer(f)
            fwriter.writerow(firstline)
            fnamelist = []
            for fname in os.listdir(bbox_dir):
                if fname.endswith('_pbb.npy'):
                    fnamelist.append(fname)
                    # print fname
                    # for row in convertcsv(fname, bbox_dir, k):
                        # fwriter.writerow(row)
            # # return
            print((len(fnamelist)))
            predannolist = []
            for fname in tqdm(fnamelist):
                predannolist.append(convertcsv(fname, result_dir=bbox_dir, detp=detpthresh))
            # predannolist = p.map(functools.partial(convertcsv, result_dir=bbox_dir, detp=detpthresh), fnamelist)
            # print len(predannolist), len(predannolist[0])
            for predanno in predannolist:
                # print predanno
                for row in predanno:
                    # print row
                    fwriter.writerow(row)
            f.close()


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(
            FROCGTList):  # Handle border case when there are no false positives and ROC analysis give nan values.
        print("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def evaluateCAD(seriesUIDs, results_path, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False, numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_path: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    nodOutputfile = open(os.path.join(outputDir, 'CADAnalysis.txt'), 'a')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = readCSV(results_path)

    allCandsCAD = {}

    for seriesuid in tqdm(seriesUIDs):

        # collect candidates from result file
        nodules = {}
        header = results[0]

        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(list(nodules.keys())) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.items():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.items():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2

        # print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules

    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(list(candidates.keys()))

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
                diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)

            found = False
            noduleMatches = []

            if IOU_TH is not None:
                best_score = 0
                for key, candidate in candidates.items():
                    x2 = float(candidate.coordX)
                    y2 = float(candidate.coordY)
                    z2 = float(candidate.coordZ)
                    d2 = float(candidate.diameter_mm)

                    box0 = np.array([x, y, z, diameter])
                    box1 = np.array([x2, y2, z2, d2])
                    score = iou(box0, box1)
                    if score >= IOU_TH:
                        if score > best_score:
                            found = True
                            best_score = score
                            noduleMatches.append(candidate)
                            if key not in list(candidates2.keys()):
                                print(
                                    "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (
                                    str(candidate.id), seriesuid, str(noduleAnnot.id)))
                            else:
                                del candidates2[key]
                        else:  # an excluded nodule
                            if key in list(candidates2.keys()):
                                irrelevantCandidates += 1
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                                seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id),
                                float(candidate.CADprobability)))
                                del candidates2[key]
            else:

                for key, candidate in candidates.items():
                    x2 = float(candidate.coordX)
                    y2 = float(candidate.coordY)
                    z2 = float(candidate.coordZ)
                    dist = np.power(x - x2, 2.) + np.power(y - y2, 2.) + np.power(z - z2, 2.)
                    if dist < radiusSquared:
                        if (noduleAnnot.state == "Included"):
                            found = True
                            noduleMatches.append(candidate)
                            if key not in list(candidates2.keys()):
                                print(
                                    "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (
                                    str(candidate.id), seriesuid, str(noduleAnnot.id)))
                            else:
                                del candidates2[key]
                        elif (noduleAnnot.state == "Excluded"):  # an excluded nodule
                            if bOtherNodulesAsIrrelevant:  # delete marks on excluded nodules so they don't count as false positives
                                if key in list(candidates2.keys()):
                                    irrelevantCandidates += 1
                                    ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                                    seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id),
                                    float(candidate.CADprobability)))
                                    del candidates2[key]
            if len(noduleMatches) > 1:  # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            if noduleAnnot.state == "Included":
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0)
                    FROCProbList.append(float(maxProb))
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1
                else:
                    candFNs += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.items():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (
            seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id),
            float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(
            FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write(
        "    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write(
        "    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)

    if performBootstrapping:
        fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(FROCGTList, FROCProbList,
                                                                                 FPDivisorList, seriesUIDs, excludeList,
                                                                                 numberOfBootstrapSamples=numberOfBootstrapSamples,
                                                                                 confidence=confidence)

    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))

    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

    sens_itp = np.interp(fps_itp, fps, sens)
    frvvlu = 0
    nxth = 0.125
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            frvvlu += ss
            nxth *= 2
        if abs(nxth - 16) < 1e-5: break
    print((frvvlu / 7, nxth))
    print((sens_itp[fps_itp == 0.125] + sens_itp[fps_itp == 0.25] + sens_itp[fps_itp == 0.5] + sens_itp[fps_itp == 1] +
           sens_itp[fps_itp == 2] \
           + sens_itp[fps_itp == 4] + sens_itp[fps_itp == 8]))
    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':')  # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':')  # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        # plt.ylim(0.5, 1)
        plt.ylim(0.0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))

        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        # ax.yaxis.set_ticks(np.arange(0.5, 1, 0.1))
        ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
        # ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)


def getNodule(annotation, header, state=""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]

    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]

    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]

    if not state == "":
        nodule.state = state

    return nodule

def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0

    for seriesuid in seriesUIDs:
        # print 'adding nodule annotations: ' + seriesuid

        nodules = []
        numberOfIncludedNodules = 0

        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1

        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Excluded")
                nodules.append(nodule)

        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)

    print('Total number of included nodule annotations: ' + str(noduleCount))
    print('Total number of nodule annotations: ' + str(noduleCountTotal))
    return allNodules

def resample_pos(label, thickness, spacing, new_spacing=[1, 1, 1]):
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    resize_factor = resize_factor[::-1]
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]
    return label

def collect(seriesuids_path):
    # annotations = csvTools.readCSV(annotations_filename)
    # annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
    # seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)

    pos_df = pd.read_csv(os.path.join(data_dir, pos_label_file), dtype={"date": str})
    imageInfo = np.load(os.path.join(data_dir, info_file), allow_pickle=True)["info"]

    patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
                     for info, id in zip(imageInfo, np.arange(len(imageInfo)))}

    header = [seriesuid_label, coordX_label, coordY_label, coordZ_label, diameter_mm_label]

    seriesUIDs = np.load(seriesuids_path).tolist()
    annotations_excluded = [header]
    annotations = [header]
    for seriesuid in seriesUIDs:

        imageId = patient2Image[seriesuid]
        pstr = imageInfo[imageId]["pstr"]
        dstr = imageInfo[imageId]["date"]
        thickness = imageInfo[imageId]["sliceThickness"]
        spacing = imageInfo[imageId]["pixelSpacing"]
        existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
        pos = pos_df[existId]
        temp0 = pos[["x", "y", "z", "d"]].values

        temp0 = np.array([resample_pos(p, thickness, spacing) for p in temp0])
        # pos = pos[:, [2, 1, 0, 3]]
        for temp1 in temp0:
            annotations.append([seriesuid] + temp1.tolist())

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)

    return (allNodules, seriesUIDs)

def noduleCADEvaluation(seriesuids_path):


    (allNodules, seriesUIDs) = collect(seriesuids_path)

    bPerformBootstrapping = False
    bNumberOfBootstrapSamples = 1000
    bConfidence = 0.95


    evaluateCAD(seriesUIDs, results_path, outputDir, allNodules,
                os.path.splitext(os.path.basename(results_path))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)


if __name__ == '__main__':

    detp = [0]
    nmsthresh = [0.1]
    # IOU_TH = 0.2
    IOU_TH = None
    # IOU_TH = None
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210121-180624/"   ## fine-tuned on methodist data
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210121-225702/"    ## trained on lunaRaw
    # result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector/results/res18-20210126-011543"    ## trained on luna (pretrained)
    result_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector_ben/results/res18-20210209-104946/"    ## trained on masked methodist data

    outputDir = result_dir
    getcsv(detp, nmsthresh, "masked")


    data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    pos_label_file = "pos_labels.csv"
    info_file = "CTinfo.npz"

    seriesuids_path = os.path.join(result_dir, "bbox/namelist.npy")
    # result_file = "luna_IOU0.2_detp0_nms0.1.csv"
    result_file = "masked_detp0_nms0.1.csv"
    results_path = os.path.join(result_dir, "bbox", result_file)
    noduleCADEvaluation(seriesuids_path)

    print("")