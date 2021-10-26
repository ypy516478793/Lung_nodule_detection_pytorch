"""
Incidental train:
    python detect.py \
        -d=methodistFull --test=False --gpu="2,3" --save-dir=CODETEST \
        --resume="../detector_ben/results/PET_newMask_rs42_augRegular_5fold_2/029.ckpt" --start-epoch=0 --best_loss=0.3076
Incidental test:
    python detect.py \
        -d=methodistFull --test=True --gpu="2,3" --save-dir=CODETEST \
        --resume="../detector_ben/results/PET_newMask_rs42_augRegular_5fold_2/029.ckpt"
Incidental inference:
    python detect.py \
        -d=methodistFull --inference=True --gpu="2,3" --save-dir=CODETEST \
        --resume="../detector_ben/results/PET_newMask_rs42_augRegular_5fold_2/029.ckpt"

Low-dose LUNA16 trained model:
    -d=luna --test=False --gpu="4,5,6,7"
    --resume="../detector_ben/results/res18-20201223-115306/038.ckpt" --start-epoch=38 --best_loss=0.1206
    --resume="../detector_ben/results/res18-20210105-171908/050.ckpt" --start-epoch=50 --best_loss=0.0703
"""

import sys
sys.path.append("../")


from detector_ben.layers import acc, top_pbb
from detector_ben.utils import *
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import imgaug.augmenters as iaa
# import nibabel as nib
import numpy as np
import argparse
import torch
import time


parser = argparse.ArgumentParser(description="PyTorch DataBowl3 Detector")
parser.add_argument("--datasource", "-d", type=str, default="luna",
                    help="luna, lunaRaw, methoidstPilot, methodistFull, additional")
parser.add_argument("--data_dir", "-p", default=None, help="Data directory")
parser.add_argument("--pad_value", "-pv", default=None, help="Pad value for patching")
parser.add_argument("--model", "-m", metavar="MODEL", default="res18", help="model")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="number of data loading workers (default: 32)")
parser.add_argument("--epochs", "-e", default=100, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--best_loss", default=np.inf, type=float,
                    help="manual best loss (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=4, type=int,
                    metavar="N", help="mini-batch size (default: 16)")
parser.add_argument("--lr", "--learning-rate", default=0.01, type=float,
                    metavar="LR", help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float,
                    metavar="W", help="weight decay (default: 1e-4)")
parser.add_argument("--save-freq", default="1", type=int, metavar="S",
                    help="save frequency")
parser.add_argument("--resume", "-re", default=None, type=str, metavar="PATH",
# parser.add_argument("--resume", "-re", default="../detector/resmodel/res18fd9020.ckpt", type=str, metavar="PATH",
# parser.add_argument("--resume", default="../detector/results/res18-20201020-113114/030.ckpt",
# parser.add_argument("--resume", default="../detector_ben/results/res18-20201202-112441/026.ckpt",
# parser.add_argument("--resume", default="../detector_ben/results/res18-20201223-115306/038.ckpt",
# parser.add_argument("--resume", default="../detector_ben/results/res18-20210106-112050_incidental/001.ckpt",
#                     type=str, metavar="PATH",
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--save-dir", "-s", default='', type=str, metavar="SAVE",
                    help="directory to save checkpoint (default: none)")
parser.add_argument("--test", "-t", default=True, type=eval, metavar="TEST",
                    help="1 do test evaluation, 0 not")
parser.add_argument("--inference", "-i", default=False, type=eval,
                    help="True if run inference (no label) else False")
parser.add_argument("--testthresh", default=-3, type=float,
                    help="threshod for get pbb")
parser.add_argument("--split", default=8, type=int, metavar="SPLIT",
                    help="In the test phase, split the image to 8 parts")  # Split changed to 1 just to check.
parser.add_argument("--gpu", default="0, 1, 2, 3", type=str, metavar="N",
                    help="use gpu")
parser.add_argument("--rseed", default=None, type=int, metavar="N",
                    help="random seed for train/val/test data split")
parser.add_argument("--limit_train", default=None, type=float, metavar="N",
                    help="ratio of training size")

parser.add_argument("--mask", default=True, type=eval, help="mask lung")
parser.add_argument("--crop", default=True, type=eval, help="crop lung")

parser.add_argument("--flip", default=False, type=eval, help="flip")
parser.add_argument("--swap", default=False, type=eval, help="swap")
parser.add_argument("--scale", default=False, type=eval, help="scale")
parser.add_argument("--rotate", default=False, type=eval, help="rotate")
parser.add_argument("--contrast", default=False, type=eval, help="contrast")
parser.add_argument("--bright", default=False, type=eval, help="bright")
parser.add_argument("--sharp", default=False, type=eval, help="sharp")
parser.add_argument("--splice", default=False, type=eval, help="splice")

parser.add_argument("--kfold", default=None, type=int, help="number of kfold for train_val")
parser.add_argument("--split_id", default=None, type=int, help="split id when use kfold")

parser.add_argument("--n_test", default=1, type=int, metavar="N",
                    help="number of gpu for test")
parser.add_argument("--train_patience", type=int, default=20,
                    help="If the validation loss does not decrease for this number of epochs, stop training")
parser.add_argument("--save_interval", type=int, default=10, help="save interval for pytorch model")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import DataParallel

def merge_args(config, args):
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.pad_value is not None:
        config.PAD_VALUE = args.pad_value
    return config

def main():
    ## Set gpu resources
    torch.manual_seed(0)

    ## Datasource (config)
    if args.datasource == "lunaRaw":
        # Dataset = datald.lunaRaw
        from dataLoader.lunaRaw import LunaRaw, LunaConfig
        config = LunaConfig()
        Dataset = LunaRaw
    elif args.datasource == "luna":
        # Dataset = datald.luna
        from dataLoader.luna import Luna, LunaConfig
        config = LunaConfig()
        Dataset = Luna
    elif args.datasource == "methodistFull":
        from dataLoader.methodistFull import MethodistFull, IncidentalConfig
        config = IncidentalConfig()
        config.SPLIT_SEED = args.rseed
        config.LIMIT_TRAIN = args.limit_train
        config.MASK_LUNG = args.mask
        config.CROP_LUNG = args.crop
        config.AUGTYPE["flip"] = args.flip
        config.AUGTYPE["swap"] = args.swap
        config.AUGTYPE["scale"] = args.scale
        config.AUGTYPE["rotate"] = args.rotate
        config.AUGTYPE["contrast"] = args.contrast
        config.AUGTYPE["bright"] = args.bright
        config.AUGTYPE["sharp"] = args.sharp
        config.AUGTYPE["splice"] = args.splice

        config.KFOLD = args.kfold
        config.SPLIT_ID = args.split_id
        Dataset = MethodistFull

    config = merge_args(config, args)

    ## Specify the save directory
    save_dir = args.save_dir
    if not save_dir:
        exp_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        if args.inference:
            mode_str = "inference"
        else:
            mode_str = "test" if args.test else "train"
        save_dir = os.path.join("detector_ben/results", args.model + "-" + exp_id + "-" + mode_str)
    else:
        save_dir = os.path.join("detector_ben/results", save_dir) # run exp from project root directory!!!

    # if not args.test:
    #     if os.path.exists(save_dir):
    #         choice = input("This directory {} already exist, [o] to overwrite it or [c] to continue training, "
    #                        "any other to stop.\n".format(save_dir))
    #         if choice == 'o':
    #             shutil.rmtree(save_dir)
    #             os.makedirs(save_dir)
    #         elif choice == 'c':
    #             pass
    #         else:
    #             raise SystemExit("Manually interrupted! Try another directory")

    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, "log")
    sys.stdout = Logger(logfile)


    # LOG EXPERIMENT CONFIGURATION
    bind = lambda x: "--{:s}={:s}".format(str(x[0]), str(x[1]))
    print("=" * 100)
    print("Running at: {:s}".format(str(datetime.now())))
    print("Working in directory: {:s}\n".format(save_dir))
    print("Run experiments: ")
    print("python {:s}".format(" ".join(sys.argv)))
    print("Full arguments: ")
    print("{:s}\n".format(" ".join([bind(i) for i in vars(args).items()])))


    config.display()

    ## Set writer
    global writer
    writer = SummaryWriter(os.path.join(save_dir, "runs/"))
    best_loss = args.best_loss
    # writer.add_graph(net, (torch.zeros(2, 1, 96, 96, 96), torch.zeros(2, 3, 24, 24, 24)))

    ## Construct model
    if args.model == "res18":
        import detector_ben.res18 as model
        net, loss, get_pbb = model.get_model(config)
    net = net.cuda()
    loss = loss.cuda()
    net = DataParallel(net)


    ## Load saved model
    start_epoch = args.start_epoch
    if args.resume:
        try:
            model_path = args.resume
            checkpoint = torch.load(model_path)
        except:
            model_list = [m for m in os.listdir(args.resume) if m.endswith("ckpt")]
            from natsort import natsorted
            latest_model = natsorted(model_list)[-1]
            model_path = os.path.join(args.resume, latest_model)
            checkpoint = torch.load(model_path)
        state_dict = checkpoint["state_dict"]
        try:
            net.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k  # add "module." for dataparallel
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        print("Load successfully from " + model_path)
    if start_epoch == 0:
        start_epoch = 1

    ## Run inference
    if args.inference:
        from dataLoader.dataBase import collate
        dataset = Dataset(config, subset="inference")
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False)

        inference(test_loader, net, get_pbb, save_dir, config)
        return

    ## Run test
    if args.test == 1:
        from dataLoader.dataBase import collate
        dataset = Dataset(config, subset="test")
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False)

        test(test_loader, net, get_pbb, save_dir, config)

    ## Run train
    else:

        optimizer = torch.optim.SGD(
            net.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)

        dataset = Dataset(config, subset="train")
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
        dataset = Dataset(config, subset="val")
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

        patience_train = 0
        for epoch in range(start_epoch, args.epochs + 1):
            print("Start epoch {:d}!".format(epoch))
            net = train(train_loader, net, loss, epoch, optimizer, get_lr)
            best_loss, patience_train = validate(val_loader, net, loss, epoch, save_dir, best_loss, patience_train, args.save_interval)


def train(data_loader, net, loss, epoch, optimizer, get_lr):
    start_time = time.time()
    saved_model_list = []
    net.train()
    lr = get_lr(epoch, args)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    metrics = []

    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    for i, (data, label, coord, target) in enumerate(tqdm(data_loader)):
        # if epoch == args.start_epoch + 1:
        #     for j in range(len(data)):
        #         if not torch.isnan(target[j][0]):
        #             fig = stack_nodule(data[j, 0].numpy(), target[j].numpy())
        #             writer.add_figure("training images",
        #                               fig, global_step=i)
        s_time = time.time()
        data = Variable(torch.FloatTensor(data).cuda(async=True))
        label = Variable(torch.FloatTensor(label).cuda(async=True))
        coord = Variable(torch.FloatTensor(coord).cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, label)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data.item()
        metrics.append(loss_output)
        e_time = time.time()
        t = e_time - s_time

        if isinstance(loss_output[6], torch.Tensor):
            loss_output[6] = loss_output[6].data.item()
        if isinstance(loss_output[8], torch.Tensor):
            loss_output[8] = loss_output[8].data.item()
        loss_output = np.array(loss_output)
        print("{}: EPOCH{:03d} Itr{}/{} ({:.2f}s/itr) Train: acc {:3.2f}, tpr {:d}/{:d}={:3.2f}, tnr {:d}/{:d}={:3.2f}, "
              "loss {:2.4f}, classify loss {:2.4f}, regress loss {:2.4f}, {:2.4f}, {:2.4f}, {:2.4f}".format(
            datetime.now(), epoch, i + 1, len(data_loader), t,
            100.0 * np.sum(loss_output[[6, 8]]) / np.sum(loss_output[[7, 9]]),
            int(loss_output[6]), int(loss_output[7]), 100.0 * np.sum(loss_output[6]) / np.max([np.sum(loss_output[7]), 1]),
            int(loss_output[8]), int(loss_output[9]), 100.0 * np.sum(loss_output[8]) / np.sum(loss_output[9]),
            loss_output[0], loss_output[1], loss_output[2], loss_output[3], loss_output[4], loss_output[5])
        )

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print("Epoch %03d (lr %.5f)" % (epoch, lr))
    print("Train: acc %3.2f, tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f" % (
        100.0 * np.sum(metrics[:, [6, 8]]) / np.sum(metrics[:, [7, 9]]),
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print("loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f" % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

    writer.add_scalar('Train/acc', 100.0 * np.sum(metrics[:, [6, 8]]) / np.sum(metrics[:, [7, 9]]), epoch)
    writer.add_scalar('Train/tpr', 100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]), epoch)
    writer.add_scalar('Train/tnr', 100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]), epoch)
    writer.add_scalar('Train/loss', np.mean(metrics[:, 0]), epoch)
    writer.add_scalar('Train/classify loss', np.mean(metrics[:, 1]), epoch)
    writer.add_scalar('Train/regress loss', np.mean(metrics[:, 2]), epoch)

    return net


def validate(data_loader, net, loss, epoch, save_dir, best_loss, patience_train, save_interval):
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, label, coord, target) in enumerate(tqdm(data_loader)):
        # if epoch == args.start_epoch + 1:
        #     for j in range(len(data)):
        #         if not torch.isnan(target[j][0]):
        #             fig = stack_nodule(data[j, 0].numpy(), target[j].numpy())
        #             writer.add_figure("validation images",
        #                               fig, global_step=epoch * len(data_loader) + i)

        with torch.no_grad():
            data = Variable(data.cuda(async=True))
            label = Variable(label.cuda(async=True))
            coord = Variable(coord.cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, label, train=False)

        loss_output[0] = loss_output[0].data.item()
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print("Validation: acc %3.2f, tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f" % (
        100.0 * np.sum(metrics[:, [6, 8]]) / np.sum(metrics[:, [7, 9]]),
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print("loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f" % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

    writer.add_scalar('Val/acc', 100.0 * np.sum(metrics[:, [6, 8]]) / np.sum(metrics[:, [7, 9]]), epoch)
    writer.add_scalar('Val/tpr', 100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]), epoch)
    writer.add_scalar('Val/tnr', 100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]), epoch)
    writer.add_scalar('Val/loss', np.mean(metrics[:, 0]), epoch)
    writer.add_scalar('Val/classify loss', np.mean(metrics[:, 1]), epoch)
    writer.add_scalar('Val/regress loss', np.mean(metrics[:, 2]), epoch)

    if epoch % save_interval == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        checkpoint = os.path.join(save_dir, "%03d.ckpt" % epoch)
        torch.save({
            "epoch": epoch,
            "save_dir": save_dir,
            "state_dict": state_dict,
            "args": args},
            checkpoint)

    loss_mean = np.mean(metrics[:, 0])
    if loss_mean < best_loss:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        checkpoint = os.path.join(save_dir, "best_%03d.ckpt" % epoch)
        torch.save({
            "epoch": epoch,
            "save_dir": save_dir,
            "state_dict": state_dict,
            "args": args},
            checkpoint)

        print("=========================EPOCH_{} loss decrease from {:.4f} to {:.4f}=========================".format(
              epoch, best_loss, loss_mean))
        print("=========================EPOCH_{} Saved checkpoing=========================".format(
              epoch))
        patience_train = 0
        best_loss = loss_mean
    else:
        print("==========================EPOCH_{} loss {:.4f} doesn't decrease from {:.4f}=========================".format(
              epoch, loss_mean, best_loss))
        patience_train = patience_train + 1
        if patience_train >= args.train_patience:
            print("=========================EPOCH_{} loss {:.4f} doesn't decrease for {} epochs, stopped!".format(
                  epoch, best_loss, patience_train))
            sys.exit()
    return best_loss, patience_train

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, "bbox")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw, imgs) in enumerate(data_loader):

        print
        print("I am at iteration " + str(i_name))

        s = time.time()
        data1 = [np.asarray(d, np.float32) for d in data]
        print("Shape of input: " + str(np.array(data1).shape))
        target = [np.asarray(t, np.float32) for t in target]
        print("TARGET IS: " + str(target))
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split("/")[-1].split("_clean")[0].rstrip(".npz")   # .split("-")[0]  wentao change
        namelist.append(name)
        data = data[0][0]
        coord = coord[0][0]
        imgs = imgs[0]
        isfeat = False
        #
        #
        # splitfun = split_comber.split
        # combinefun = split_comber.combine
        # z, h, w = data.size(2), data.size(3), data.size(4)
        # print(data.size())
        # data = splitfun(data, max_stride=config.MAX_STRIDE, margin=config.MARGIN)
        # data = Variable(data.cuda(async=True), volatile=True, requires_grad=False)
        # splitlist = range(0, args.split + 1, n_per_run)
        # outputlist = []
        # featurelist = []
        # for i in range(len(splitlist) - 1):
        #     if isfeat:
        #         output, feature = net(data[splitlist[i]:splitlist[i + 1]])
        #         featurelist.append(feature)
        #     else:
        #         output = net(data[splitlist[i]:splitlist[i + 1]])
        #     output = output.data.cpu().numpy()
        #     outputlist.append(output)
        #
        # output = np.concatenate(outputlist, 0)
        # output = combinefun(output, z / config.STRIDE, h / config.STRIDE, w / config.STRIDE)
        # if isfeat:
        #     feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
        #     feature = combinefun(feature, z / config.STRIDE, h / config.STRIDE, w / config.STRIDE)










        # if "output_feature" in config:
        #     if config["output_feature"]:
        #         isfeat = True
        n_per_run = args.n_test

        splitlist = list(range(0, len(data) + 1, n_per_run))  # Check if n_per_run is doing something to the splitlist.

        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        print("The splitlist is: " + str(splitlist))
        for i in range(len(splitlist) - 1):
            with torch.no_grad():

                # input= Variable(data[splitlist[i]]).cuda()
                # inputcoord = Variable(coord[splitlist[i]]).cuda()

                input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())  ## Shape is (6,4,......)
            del output
        # print()
        print("The shape of outputlist is:  " + str(np.array(outputlist).shape))
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        # if isfeat:
        #     feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
        #     feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = args.testthresh  # -8 #-3
        print("pbb thresh", thresh)
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + "_feature.npy"), feature_selected)

        delete_row = []
        for row, pi in enumerate(pbb):
            if np.any(pi[1:4] >= imgs[0].shape):
                delete_row.append(row)
        pbb = np.delete(pbb, delete_row, 0)

        tp, fp, fn, _ = acc(pbb, lbb, config.CONF_TH, 0.5, 0.5)
        tps = np.array(tp)
        fps = np.array(fp)
        print("The lengths of TRUE POSITIVES and FALSE POSITIVES are: ")
        print(len(tp), len(fp))

        print("TP: ")
        print(tps)
        for tp_i in tps:
            fig = stack_nodule(imgs[0], tp_i[1:5], prob=tp_i[0], show_every=1)
            writer.add_figure("TP images",
                              fig, global_step=i_name)

        print("FP: ")
        print(fps[:2])
        if len(fps) > 0:
            fig = stack_nodule(imgs[0], fps[0][1:5], prob=fps[0][0], show_every=1)
            writer.add_figure("FP images",
                              fig, global_step=i_name)

        if not os.path.exists(os.path.join(save_dir, name + "_tp.txt")):
            with open(os.path.join(save_dir, name + "_tp.txt"), "a+") as f:
                f.write("Patient name is: " + str(name))
                f.write("\n")
                f.write("\n")
                f.write("The predicted bounding boxes are:\n")
                for bb in range(len(tp)):
                    f.write(str(tp[bb][1:5]) + "\n")
                f.write("\n")
                f.write("\n")
                f.write("\n")
                f.write("The ground truth bounding boxes are: \n")
                for box in range(len(target)):
                    f.write(str(target[box]) + "\n")
            f.close()
        print("The true positive bounding boxes are:  ")
        print(tp)
        print("The false positive bounding boxes are: ")
        print(fp)
        print([i_name, name])
        e = time.time()
        np.save(os.path.join(save_dir, name + "_pbb.npy"), pbb)
        np.save(os.path.join(save_dir, name + "_lbb.npy"), lbb)
    np.save(os.path.join(save_dir, "namelist.npy"), namelist)
    end_time = time.time()
    print("elapsed time is %3.2f seconds" % (end_time - start_time))

def inference(data_loader, net, get_pbb, save_dir, config):
    from show_results import plot_bbox
    start_time = time.time()
    save_dir = os.path.join(save_dir, "preds")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()

    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw, imgs, infos) in enumerate(data_loader):

        print
        print("I am at iteration " + str(i_name))

        s = time.time()
        data1 = [np.asarray(d, np.float32) for d in data]
        print("Shape of input: " + str(np.array(data1).shape))
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split("/")[-1].split("_clean")[0].rstrip(".npz")  # .split("-")[0]  wentao change
        namelist.append(name)
        print("Patient MRN-date: ", name)

        # if name != "1.3.6.1.4.1.14519.5.2.1.6279.6001.340158437895922179455019686521_image":
        #     continue


        data = data[0][0]
        coord = coord[0][0]
        imgs = imgs[0]
        infos = infos[0].tolist()
        isfeat = False
        n_per_run = args.n_test

        splitlist = list(range(0, len(data) + 1, n_per_run))  # Check if n_per_run is doing something to the splitlist.

        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        print("The splitlist is: " + str(splitlist))
        for i in range(len(splitlist) - 1):
            with torch.no_grad():

                # input= Variable(data[splitlist[i]]).cuda()
                # inputcoord = Variable(coord[splitlist[i]]).cuda()

                input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())  ## Shape is (6,4,......)
        # print()
        print("The shape of outputlist is:  " + str(np.array(outputlist).shape))
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        # if isfeat:
        #     feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
        #     feature = split_comber.combine(feature, sidelen)[..., 0]

        os.makedirs(os.path.join(save_dir, name), exist_ok=True)
        thresh = args.testthresh  # -8 #-3
        print("pbb thresh", thresh)
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + "_feature.npy"), feature_selected)

        pbb = top_pbb(pbb, 5, config)

        ori_str = "_ori" if config.ORIGIN_SCALE else ""
        pbb_infer = deepcopy(pbb)
        if config.ORIGIN_SCALE:
            thickness, spacing = infos["sliceThickness"], infos["pixelSpacing"]
            img_infer = invert_image(imgs[0], thickness, spacing)[0]
            for j in range(len(pbb)):
                pbb_infer[j][1:5] = invert_pos(pbb[j][1:5], thickness, spacing)
        else:
            img_infer = imgs[0]

        delete_row = []
        for row, pi in enumerate(pbb_infer):
            if np.any(pi[1:4] >= img_infer.shape):
                delete_row.append(row)
        pbb_infer = np.delete(pbb_infer, delete_row, 0)

        n_show = np.min([len(pbb_infer), 10])
        for j in range(n_show):
            fig = stack_nodule(img_infer, pbb_infer[j][1:5], prob=pbb_infer[j][0], show_every=1)
            plt.savefig(os.path.join(save_dir, name, "pred{:s}_{:d}.png".format(ori_str, j)),
                        bbox_inches="tight", dpi=200)
            plt.close(fig)
            plot_bbox(os.path.join(save_dir, name, "pred{:s}_{:d}".format(ori_str, j)), img_infer, pbb_infer[j][1:5], show=False)

        # save into nifti format
        img_infer_xyz = np.transpose(img_infer, [2, 1, 0])
        imgNifti = nib.Nifti1Image(img_infer_xyz, np.eye(4))
        nib.save(imgNifti, os.path.join(save_dir, name, "recover_CT_testsample{:s}.nii.gz".format(ori_str)))

        np.save(os.path.join(save_dir, name, "pbb{:s}.npy".format(ori_str)), pbb_infer)
        if not os.path.exists(os.path.join(save_dir, name, "pbb{:s}.txt".format(ori_str))):
            with open(os.path.join(save_dir, name, "pbb{:s}.txt".format(ori_str)), "a+") as f:
                f.write("Patient id (MRN - date): " + str(name))
                f.write("\n\n")
                f.write("Bounding boxes: [z, y, x, d]\n")
                f.write("z: slice index \n")
                f.write("y: row index \n")
                f.write("x: column index \n")
                f.write("d: diameter of the nodule \n")
                f.write("\n")
                f.write("The predicted bounding boxes are:\n")
                for j in range(n_show):
                    f.write("{:d}: ".format(j) + str(pbb_infer[j][1:5]) + "\n")
                f.write("\n")
            f.close()

    np.save(os.path.join(save_dir, "namelist.npy"), namelist)
    end_time = time.time()
    print("elapsed time is %3.2f seconds" % (end_time - start_time))


def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config["max_stride"],margin)
    data = Variable(data.cuda(async= True), volatile = True,requires_grad=False)
    splitlist = range(0,args.split+1,n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output,feature = net(data[splitlist[i]:splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)

    output = np.concatenate(outputlist,0)
    output = combinefun(output, z / config["stride"], h / config["stride"], w / config["stride"])
    if isfeat:
        feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])
        feature = combinefun(feature, z / config["stride"], h / config["stride"], w / config["stride"])
        return output,feature
    else:
        return output
if __name__ == "__main__":
    main()
