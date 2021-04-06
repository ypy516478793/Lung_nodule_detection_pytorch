from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


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

def main(root_dir, kfold):
    # compute FROC
    sens_itp_list = []
    for i in range(kfold):
        result_dir = root_dir + "_{:d}fold_{:d}".format(kfold, i)
        result_file = "froc_gt_prob_vectors_masked_cropped_detp0.0_nms0.1.csv"
        result_path = os.path.join(result_dir, result_file)
        result = np.genfromtxt(result_path, delimiter=",")
        FROCGTList, FROCProbList = result[:, 0], result[:, 1]

        seriesuids_path = os.path.join(result_dir, "bbox/namelist.npy")
        seriesUIDs = np.load(seriesuids_path).tolist()
        excludeList = np.zeros_like(FROCProbList)
        false_neg_ids = np.where(np.array(FROCProbList) == -1000000000)[0]
        excludeList[false_neg_ids] = True

        fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)

        # if performBootstrapping:
        #     fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(FROCGTList, FROCProbList,
        #                                                                              FPDivisorList, seriesUIDs, excludeList,
        #                                                                              numberOfBootstrapSamples=numberOfBootstrapSamples,
        #                                                                              confidence=confidence)

        # # Write FROC curve
        # with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        #     for i in range(len(sens)):
        #         f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
        #
        # # Write FROC vectors to disk as well
        # with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        #     for i in range(len(FROCGTList)):
        #         f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

        fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
        sens_itp = np.interp(fps_itp, fps, sens)
        sens_itp_list.append(sens_itp)
    sens_itp_mean = np.mean(np.array(sens_itp_list), axis=0)
    sens_itp_std = np.std(np.array(sens_itp_list), axis=0)
    plot_froc_average(fps_itp, sens_itp_mean, sens_itp_std)


def plot_froc_average(fps_itp, sens_itp, sens_itp_std):
    frvvlu = 0
    nxth = 0.125
    fps_sens_list = []
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            frvvlu += ss
            nxth *= 2
            fps_sens_list.append(ss)
        if abs(nxth - 16) < 1e-5: break
    # print((frvvlu / 7, nxth))
    print("Average precision: ", np.mean(fps_sens_list))
    assert nxth == 16, "nxth is not 16!"
    print("Precision @ 0.125/0.25/0.5/1/2/4/8", fps_sens_list)
    with open(os.path.join(outputDir, "froc_mAP.csv"), "w") as f:
        f.write("mAP: {:f}\n".format(np.mean(fps_sens_list)))
        f.write("APs: {:}".format(fps_sens_list))
    # print((sens_itp[fps_itp == 0.125] + sens_itp[fps_itp == 0.25] + sens_itp[fps_itp == 0.5] + sens_itp[fps_itp == 1] + \
    #        sens_itp[fps_itp == 2] + sens_itp[fps_itp == 4] + sens_itp[fps_itp == 8]))
    # if performBootstrapping:
    #     # Write mean, lower, and upper bound curves to disk
    #     with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
    #         f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
    #         for i in range(len(fps_bs_itp)):
    #             f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    # else:
    #     fps_bs_itp = None
    #     sens_bs_mean = None
    #     sens_bs_lb = None
    #     sens_bs_up = None

    # create FROC graphs
    graphTitle = str("")
    fig1 = plt.figure()
    ax = plt.gca()
    clr = 'b'
    plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
    # if performBootstrapping:
    plt.plot(fps_itp, sens_itp, color=clr, ls='--')
    plt.plot(fps_itp, sens_itp + sens_itp_std, color=clr, ls=':')  # , label = "lb")
    plt.plot(fps_itp, sens_itp - sens_itp_std, color=clr, ls=':')  # , label = "ub")
    ax.fill_between(fps_itp, sens_itp - sens_itp_std, sens_itp + sens_itp_std, facecolor=clr, alpha=0.05)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="merge froc curve script")
    parser.add_argument('-bs', '--bootstrap', type=eval, help='whether to apply boot strapping', default=False)
    parser.add_argument('-s', '--save_dir', type=str, help='save directory', default=None)
    parser.add_argument('-r', '--root_dir', type=str, help='root directory', default=None)
    parser.add_argument('-n', '--model_name', type=str, help='model name', default="average")
    parser.add_argument('-b', '--blog', type=eval, help='whether to use bi-log plot', default=True)
    parser.add_argument('-k', '--kfold', type=int, help='number of kfold', default=None)
    args = parser.parse_args()

    performBootstrapping = args.bootstrap
    outputDir = args.save_dir
    root_dir = args.root_dir
    CADSystemName = args.model_name
    bLogPlot = args.blog
    kfold = args.kfold
    FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
    FROC_maxX = 8
    os.makedirs(outputDir, exist_ok=True)

    # root_dir = "/home/cougarnet.uh.edu/pyuan2/Projects2021/Lung_nodule_detection_pytorch/detector_ben/results/worker32_batch8_kim_masked_crop_nonPET_lr001_rs128_limit0.2"

    main(root_dir, kfold)