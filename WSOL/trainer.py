# Original Code: https://github.com/zhangyongshun/resnet_finetune_cub

import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
from models.models_for_cub import ResNet
from cub import cub200
import os
import matplotlib.pyplot as plt
import shutil
from utils.Config import Config
from CDI_transforms import Interlacing 
from Cutout.util.cutout import Cutout




def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    # cut_w = np.int(W * cut_rat)
    # cut_h = np.int(H * cut_rat)
   
    # Use int() or np.int32() instead of deprecated np.int
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class NetworkManager(object):
    def __init__(self, options, path):
        self.options = options
        self.path = path
        self.device = options['device']

        print('Starting to prepare network and data...')

        self.net = nn.DataParallel(self._net_choice(self.options['net_choice'])).to(self.device)
        #self.net.load_state_dict(torch.load('/home/zhangyongshun/se_base_model/model_save/ResNet/backup/epoch120/ResNet50-finetune_fc_cub.pkl'))
        print('Network is as follows:')
        print(self.net)
        #print(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(
            self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay']
        )
        self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=30, gamma=0.1)
        #self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4
        #)

        train_transform_list = [
            transforms.RandomResizedCrop(self.options['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Interlacing(probability=.8,entre=0.5), 
            
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            #Cutout(n_holes=1, length=8),
        ]
        test_transforms_list = [
            transforms.Resize(int(self.options['img_size']/0.875)),
            transforms.CenterCrop(self.options['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
        train_data = cub200(self.path['data'], train=True, transform=transforms.Compose(train_transform_list))
        test_data = cub200(self.path['data'], train=False, transform=transforms.Compose(test_transforms_list))
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    def train(self):
        epochs  = np.arange(1, self.options['epochs']+1)
        test_acc = list()
        train_acc = list()
        print('Training process starts:...')
        if torch.cuda.device_count() > 1:
            print('More than one GPU are used...')
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc')
        print('-'*50)
        best_acc = 0.0
        best_epoch = 0
        self.net.train(True)
        for epoch in range(self.options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            for imgs, labels in self.train_loader:
                self.solver.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                cutmix=True
                if cutmix:
                    r = np.random.rand(1)
                    if  r < 0.5:
                        # generate mixed sample
                        lam = np.random.beta(1., 1.)
                        rand_index = torch.randperm(imgs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                        # compute output
                        output = self.net(imgs)
                        #loss = criterion(output, target_a)
                        loss = self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)
                    else:
                        output = self.net(imgs)
                        loss = self.criterion(output, labels)

                else:




                    output = self.net(imgs)
                    loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()

            train_acc_epoch = num_correct.detach().cpu().numpy()*100 / num_total
            avg_train_loss_epoch  = sum(train_loss_epoch)/len(train_loss_epoch)
            test_acc_epoch = self._accuracy()
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            self.schedule.step()
            if test_acc_epoch>best_acc:
                best_acc = test_acc_epoch
                best_epoch = epoch+1
                print('*', end='')
                state = {
                'epoch': epoch + 1,
                'state_dict': self.net.state_dict(),
                'optimizer': self.solver.state_dict(),
                'best_acc': best_acc
                    }
            
            # Save checkpoint
                checkpoint_path = os.path.join(
                self.path['model_save'], 
                f'{self.options["net_choice"]}{self.options["model_choice"]}_best.pth.tar'
                )
                self.save_checkpoint(state, is_best=True, filename=checkpoint_path)
                #torch.save(self.net.state_dict(), os.path.join(self.path['model_save'], self.options['net_choice'], self.options['net_choice']+str(self.options['model_choice'])+'.pkl'))
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch+1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))
        plt.figure()
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')

        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title(self.options['net_choice']+str(self.options['model_choice']))
        plt.savefig(self.options['net_choice']+str(self.options['model_choice'])+'.png')

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                _, pred = torch.max(output, 1)
                num_acc += torch.sum(pred==labels.detach_())
                num_total += labels.size(0)
        return num_acc.detach().cpu().numpy()*100/num_total

    # def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    #     torch.save(state, filename)
    #     if is_best:
    #         shutil.copyfile(filename, 'model_best.pth.tar')
    # def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
    #     torch.save(state, filename)
    #     if is_best:
    #         best_model_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
    #         shutil.copyfile(filename, best_model_path)
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        # Save the entire state dictionary of the DataParallel model
        if isinstance(self.net, nn.DataParallel):
            state['state_dict'] = self.net.module.state_dict()
        else:
            state['state_dict'] = self.net.state_dict()
        
        torch.save(state, filename)
        if is_best:
            best_model_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
            shutil.copyfile(filename, best_model_path)

    def _net_choice(self, net_choice):
        if net_choice=='ResNet':
            return ResNet(pre_trained=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice=='ResNet_ED':
            return ResNet_ED(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_SE':
            return ResNet_SE(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_self':
            return ResNet_self(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])

    def adjust_learning_rate(optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
