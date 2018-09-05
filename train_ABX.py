import os
import argparse
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset_ABX import Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--device_ids", type=int, default=0, help="move to GPU")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--A", type=str, default="train_X_A", help='path of files targeted')
parser.add_argument("--B", type=str, default="train_X_B", help='path of files to process')
parser.add_argument("--val_A", type=str, default="val_X_A", help='path of files targeted')
parser.add_argument("--val_B", type=str, default="val_X_B", help='path of files to process')

parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="X", help='Super-resolution (S) or denoise training (N)')

opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    start = time.time()
    dataset_train = Dataset(train=True, data_path_A=opt.A, data_path_B=opt.B, data_path_val_A=opt.val_A, data_path_val_B=opt.val_B,  patch_size_dn=30, patch_size_sr=120, stride=5, aug_times=2, if_reseize=True)
    dataset_val = Dataset(train=False, data_path_A=opt.A, data_path_B=opt.B, data_path_val_A=opt.val_A, data_path_val_B=opt.val_B,  patch_size_dn=30, patch_size_sr=120, stride=5, aug_times=2, if_reseize=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n\n" % int(len(dataset_train)))
    end = time.time()
    print (round(end - start, 7))
    
    # Build model
    net_dn = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net_dn.apply(weights_init_kaiming)
    criterion_dn = nn.MSELoss(size_average=False)

    # Build model
    net_sr = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net_sr.apply(weights_init_kaiming)
    criterion_sr = nn.MSELoss(size_average=False)


    # Move to GPU
    device_ids = [opt.device_ids] # we will deal with this later
    model_dn = nn.DataParallel(net_dn, device_ids=device_ids).cuda()
    model_sr = nn.DataParallel(net_sr, device_ids=device_ids).cuda()

    criterion_dn.cuda()
    criterion_sr.cuda()
    # Optimizer
    optimizer_dn = optim.Adam(model_dn.parameters(), lr=opt.lr)
    optimizer_sr = optim.Adam(model_sr.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    Upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear')
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer_dn.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # set learning rate for the second model
        for param_group_s in optimizer_sr.param_groups:
            param_group_s["lr"] = current_lr

        # train
        for i, data in enumerate(loader_train, 0):
            #print(Variable(data).size())
            img_A_train, img_LB_data = Variable(data[0]), Variable(data[1], requires_grad=False)
            img_L_train, img_B_train = torch.split(img_LB_data, 1, dim=1)
            #print(img_A_train.size())
            difference_dn = img_B_train - img_L_train
            img_A_train, img_L_train, img_B_train = Variable(img_A_train.cuda()), Variable(img_L_train.cuda()), Variable(img_B_train.cuda())
            difference_dn = Variable(difference_dn.cuda())
            # training step
            model_dn.train()
            model_sr.train()

            # Update super-resolution network
            model_sr.zero_grad()
            optimizer_sr.zero_grad()
            
            out_train_dn = model_dn(img_B_train)                       
            loss_dn = criterion_dn(out_train_dn, difference_dn) / (img_B_train.size()[0]*2)            
            in_train_sr = Variable(img_B_train.cuda() - out_train_dn.cuda())
            in_train_sr = Upsample_4x(in_train_sr)
            difference_sr = in_train_sr - img_A_train
            
            out_train_sr = model_sr(in_train_sr.detach())
            loss_sr = criterion_sr(out_train_sr, difference_sr) / (img_A_train.size()[0]*2)
            loss_sr.backward()
            optimizer_sr.step()
            model_sr.eval()

            # Update denoiser network
            model_dn.zero_grad()            
            optimizer_dn.zero_grad()
            out_train_dn = model_dn(img_B_train)
            loss_dn2 = criterion_dn(out_train_dn, difference_dn) / (img_B_train.size()[0]*2)
            loss_dn2.backward()
            optimizer_dn.step()
            model_dn.eval()
            

            # results            
            out_train_dn = torch.clamp(img_B_train - out_train_dn, 0., 1.)
            out_train_sr = torch.clamp(in_train_sr - out_train_sr, 0., 1.)

            psnr_train = batch_PSNR(out_train_sr, img_A_train, 1.)
            print("[epoch %d][%d/%d] loss_dn: %.4f loss_sr: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss_dn2.item(), loss_sr.item(), psnr_train))
            
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss_sr', loss_sr.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            torch.save(model_dn.state_dict(), os.path.join(opt.outf,"epoch_%d_net_dn.pth" %(epoch+1)))
            torch.save(model_sr.state_dict(), os.path.join(opt.outf,"epoch_%d_net_sr.pth" %(epoch+1)))
            
            img_A_save = torch.clamp(difference_sr, 0., 1.)
            img_A_save= img_A_save[0,:,:].cpu()
            img_A_save= img_A_save[0].detach().numpy().astype(np.float32)*255
            #print(np.amax(img_A_save))
            cv2.imwrite(os.path.join(opt.outf, "%#04dA.png" % (step)), img_A_save)

            img_B_save = torch.clamp(out_train_sr, 0., 1.)
            img_B_save= img_B_save[0,:,:].cpu()
            img_B_save= img_B_save[0].detach().numpy().astype(np.float32)*255
            #print(np.amax(img_A_save))
            cv2.imwrite(os.path.join(opt.outf, "%#04dB.png" % (step)), img_B_save)
        
        ## the end of each epoch
        model_dn.eval()
        model_sr.eval()
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val_A = torch.unsqueeze(dataset_val[k][0], 0)
            img_val_B = torch.unsqueeze(dataset_val[k][1], 0)
            img_val_A, img_val_B = Variable(img_val_A.cuda()), Variable(img_val_B.cuda())
            
            out_val_dn = model_dn(img_val_B)            
            in_val_sr = Variable(img_val_B.cuda() - out_val_dn.cuda())           
            in_val_sr = Upsample_4x(in_val_sr)
            
            out_val_dn = torch.clamp(out_val_dn, 0., 1.)
            out_val_sr = model_sr(in_val_sr)            
            out_val_sr = torch.clamp(in_val_sr - out_val_sr, 0., 1.)
            psnr_val += batch_PSNR(out_val_sr, img_val_A, 1.)

            

        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        
        # log the images
        out_train_dn = model_dn(img_B_train)
        out_train_dn = torch.clamp(img_B_train-out_train_dn, 0., 1.)
        
        in_train_sr = Variable(out_train_dn.cuda(), requires_grad=False)
        in_train_sr.resize_(img_val_A.size())      
        out_train_sr = model_sr(in_train_sr)

        Img_A = utils.make_grid(img_A_train.data, nrow=8, normalize=True, scale_each=True)
        Img_B = utils.make_grid(img_B_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train_dn.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img_A, epoch)
        writer.add_image('input image', Img_B, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)

        
        
        # save model
        torch.save(model_dn.state_dict(), os.path.join(opt.outf, 'net_dn.pth'))
        torch.save(model_sr.state_dict(), os.path.join(opt.outf, 'net_sr.pth'))


if __name__ == "__main__":
    main()
