import sys
sys.path.append('../')
# sys.path.append(".")
import argparse
import os
import time
import numpy as np
import detector.data as datald
from importlib import import_module
import shutil
from tqdm import tqdm
from detector.utils import *
import pickle
# sys.path.append('../')

from detector.split_combine import SplitComb
from sklearn.model_selection import train_test_split
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
# from config_training import config as config_training

from detector.layers import acc

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--datasource', '-d', type=str, default='luna',
                    help='luna, lunaRaw, methoidstPilot, methoidstFull, additional')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18', help='model')
parser.add_argument('--config', '-c', default='config_training', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='resmodel/res18fd9020.ckpt', type=str, metavar='PATH',
# parser.add_argument('--resume', default='/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/detector/results/res18-20201020-113114/030.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts') # Split changed to 1 just to check.
# parser.add_argument('--gpu', default='4, 5, 6, 7', type=str, metavar='N',
parser.add_argument('--gpu', default='6, 7', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=2, type=int, metavar='N',
                    help='number of gpu for test')

def listFiles(config_training):
    datarootdir = config_training["data_root_path"]
    traindatadir = config_training['train_data_path']
    valdatadir = config_training['val_data_path']
    testdatadir = config_training['test_data_path']
    datasource = args.datasource

    if datasource == "lunaRaw":
        trainfilelist = []
        for folder in os.listdir(traindatadir):
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                    trainfilelist.append(folder + '/' + f[:-4])
        valfilelist = []
        for folder in os.listdir(valdatadir):
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                    valfilelist.append(folder + '/' + f[:-4])
        testfilelist = []
        for folder in os.listdir(testdatadir):
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                    testfilelist.append(folder + '/' + f[:-4])

    elif datasource == "luna":
        trainfilelist = []
        for folder in traindatadir:
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('_clean.npy') and f[:-10] not in config_training['black_list']:
                    trainfilelist.append(folder + '/' + f[:-10])
        valfilelist = []
        for folder in valdatadir:
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('_clean.npy') and f[:-10] not in config_training['black_list']:
                    valfilelist.append(folder + '/' + f[:-10])
        testfilelist = []
        for folder in testdatadir:
            for f in os.listdir(os.path.join(datarootdir, folder)):
                if f.endswith('_clean.npy') and f[:-10] not in config_training['black_list']:
                    testfilelist.append(folder + '/' + f[:-10])

    elif datasource == "methoidstFull":
        imageInfo = np.load("/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/CTinfo.npz",
                            allow_pickle=True)["info"]
        # s = imageInfo[0]["imagePath"].find("Lung_patient")
        # filelist = [i["imagePath"][s:] for i in imageInfo]
        trainfilelist, valfilelist = train_test_split(imageInfo, test_size=0.4, random_state=42)
        valfilelist, testfilelist = train_test_split(valfilelist, test_size=0.5, random_state=42)

    return trainfilelist, valfilelist, testfilelist


def main():
    global args
    args = parser.parse_args()
    config_training = import_module(args.config)
    config_training = config_training.config
    # from config_training import config as config_training
    torch.manual_seed(0)
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    # torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        # if start_epoch == 0:
        #     start_epoch = checkpoint['epoch'] + 1
        # if not save_dir:
        #     save_dir = checkpoint['save_dir']
        # else:
        #     save_dir = os.path.join('results',save_dir)
        net.load_state_dict(checkpoint['state_dict'])
        print("Load successfully from " + args.resume)
    # else:
    if start_epoch == 0:
        start_epoch = 1
    if not save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join('results', args.model + '-' + exp_id)
    else:
        save_dir = os.path.join('results',save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test!=1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        # for f in pyfiles:
        #     shutil.copy(f,os.path.join(save_dir,f))
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = False                     #True
    net = DataParallel(net)
    datarootdir = config_training["data_root_path"]


    trainfilelist, valfilelist, testfilelist = listFiles(config_training)

   #  trainfilelist = []
   # # with open("/home/mpadmana/anaconda3/envs/DeepLung_original/luna_patient_names/luna_train_list.pkl",'rb') as f:
   #  #    trainfilelist=pickle.load(f)
   #  with open("../methodist_patient_names/methodist_train.pkl",'rb') as f:
   #
   #      trainfilelist=pickle.load(f)
   #
   #  valfilelist = []
   #  #with open("/home/mpadmana/anaconda3/envs/DeepLung_original/luna_patient_names/luna_val_list.pkl",'rb') as f:
   #   #   valfilelist=pickle.load(f)
   #  with open ("../methodist_patient_names/methodist_val.pkl",'rb') as f:
   #      valfilelist=pickle.load(f)
   #  testfilelist = []
   #  #with open("/home/mpadmana/anaconda3/envs/DeepLung_original/luna_patient_names/luna_test_list.pkl",'rb') as f:
   #   #   testfilelist=pickle.load(f)
   #  with open("../methodist_patient_names/methodist_test.pkl",'rb') as f:
   #      testfilelist=pickle.load(f)
   #  # testfilelist=['download20180608140526download20180608140500001_1_3_12_30000018060618494775800001943']

    # trainfilelist = [i.split("_")[0] for i in os.listdir(traindatadir) if i.endswith("_clean.npy")]
    # valfilelist = [i.split("_")[0] for i in os.listdir(valdatadir) if i.endswith("_clean.npy")]
    # testfilelist = [i.split("_")[0] for i in os.listdir(testdatadir) if i.endswith("_clean.npy")]

    if args.datasource == "lunaRaw":
        Dataset = datald.lunaRaw
    elif args.datasource == "luna":
        Dataset = datald.luna
    elif args.datasource == "methoidstFull":
        Dataset = datald.methodistFull

    if args.test == 1:

        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        dataset = Dataset(
            datarootdir,
            testfilelist,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size = 2,
            shuffle = False,
            num_workers = 0,
            collate_fn = datald.collate,
            pin_memory=False)

        # for i, (data, target, coord, nzhw) in enumerate(test_loader): # check data consistency
        #     if i >= len(testfilelist)/args.batch_size:
        #         break
        
        test(test_loader, net, get_pbb, save_dir,config)

        return
    #net = DataParallel(net)
    print(len(trainfilelist))
    dataset = Dataset(
        datarootdir,
        trainfilelist,
        config,
        phase = 'train')
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    dataset = Dataset(
        datarootdir,
        valfilelist,
        config,
        phase = 'val')
    val_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    # for i, (data, target, coord) in enumerate(train_loader): # check data consistency
    #     if i >= len(trainfilelist)/args.batch_size:
    #         break
    #
    # for i, (data, target, coord) in enumerate(val_loader): # check data consistency
    #     if i >= len(valfilelist)/args.batch_size:
    #         break

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 1/3: #0.5:
            lr = args.lr
        elif epoch <= args.epochs * 2/3: #0.8:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    

    for epoch in range(start_epoch, args.epochs + 1):
        print("Start epoch {:d}!".format(epoch))
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
        validate(val_loader, net, loss)

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()
    saved_model_list = []
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(tqdm(data_loader)):
        data = Variable(torch.FloatTensor(data).cuda(async = True))
        target = Variable(torch.FloatTensor(target).cuda(async = True))
        coord = Variable(torch.FloatTensor(coord).cuda(async = True))

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data.item()
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()


        # dir_checkpoint = os.path.join(save_folder, "checkpoints/")
        # if average_training_loss < best_loss:
        #     pt_name = dir_checkpoint + "CP_epoch{}.pth".format(epoch + 1)
        #     torch.save(net.state_dict(), pt_name)
        #     print("Checkpoint {} saved !".format(epoch + 1))
        #     best_loss = average_training_loss
        #     saved_model_list.append(pt_name)
        #     if len(saved_model_list) > max_to_keep:
        #         delete_model = saved_model_list.pop(0)
        #         os.remove(delete_model)

        checkpoint = os.path.join(save_dir, '%03d.ckpt' % epoch)
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            checkpoint)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print


def validate(data_loader, net, loss):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        with torch.no_grad():
            
            data = Variable(data.cuda(async = True))
            target = Variable(target.cuda(async = True))
            coord = Variable(coord.cuda(async = True))

        output = net(data, coord)
        loss_output = loss(output, target, train = False)

        loss_output[0] = loss_output[0].data.item()
        metrics.append(loss_output)    
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    print

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        
        print
        print("I am at iteration "+str(i_name))
        
        s = time.time()
        data1=[np.asarray(d, np.float32) for d in data]
        # print("Shape of input: "+str(np.array(data1).shape))
        target = [np.asarray(t, np.float32) for t in target]
        print("TARGET IS: "+ str(target))
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0] #.split('-')[0]  wentao change
        namelist.append(name)
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
         
        splitlist = range(0,len(data)+1,n_per_run)   # Check if n_per_run is doing something to the splitlist.
         
        if splitlist[-1]!=len(data):
            list(splitlist).append(len(data))
        outputlist = []
        featurelist = []
        
        print("The splitlist is: "+ str(splitlist))
        for i in range(len(splitlist)-1):   
            with torch.no_grad():
                
                #input= Variable(data[splitlist[i]]).cuda()
                #inputcoord = Variable(coord[splitlist[i]]).cuda()
                
                input = Variable(data[splitlist[i]:splitlist[i+1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]]).cuda()
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input,inputcoord)
            outputlist.append(output.data.cpu().numpy())  ## Shape is (6,4,......)
        print()
        print("The shape of outputlist is:  "+ str(np.array(outputlist).shape))    
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]
        
        thresh = args.testthresh # -8 #-3
        print('pbb thresh', thresh)
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        
        tp,fp,fn,_ = acc(pbb,lbb,0,0.5,0.5)
        print("The lengths of TRUE POSITIVES and FALSE POSITIVES are: ")
        print(len(tp),len(fp))
        
        if not os.path.exists(os.path.join(save_dir,name+'_tp.txt')):
            with open (os.path.join(save_dir,name+'_tp.txt'),'a+') as f:
                f.write("Patient name is: "+ str(name))
                f.write('\n')
                f.write('\n')
                f.write("The predicted bounding boxes are:\n")
                for bb in range(len(tp)):
                    f.write(str(tp[bb][1:5])+'\n')
                f.write('\n')
                f.write('\n')
                f.write('\n')
                f.write("The ground truth bounding boxes are: \n")
                for box in range(len(target)):
                    f.write(str(target[box])+'\n')
            f.close()
        print("The true positive bounding boxes are:  ")
        print(tp)
        print("The false positive bounding boxes are: ")
        print(fp)
        print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))

# def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
#     z, h, w = data.size(2), data.size(3), data.size(4)
#     print(data.size())
#     data = splitfun(data,config['max_stride'],margin)
#     data = Variable(data.cuda(async= True), volatile = True,requires_grad=False)
#     splitlist = range(0,args.split+1,n_per_run)
#     outputlist = []
#     featurelist = []
#     for i in range(len(splitlist)-1):
#         if isfeat:
#             output,feature = net(data[splitlist[i]:splitlist[i+1]])
#             featurelist.append(feature)
#         else:
#             output = net(data[splitlist[i]:splitlist[i+1]])
#         output = output.data.cpu().numpy()
#         outputlist.append(output)
#
#     output = np.concatenate(outputlist,0)
#     output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
#     if isfeat:
#         feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])
#         feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
#         return output,feature
#     else:
#         return output
if __name__ == '__main__':
    main()
