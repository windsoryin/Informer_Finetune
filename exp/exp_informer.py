from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Finetune,Dataset_Finetune_pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, DTW, Temporal, Score

import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
from loss.dilate_loss import dilate_loss
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.writer = None

    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':

            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers

            # model = nn.DataParallel(self.args.model)

            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'Wind_data4':Dataset_Custom,
            'gnss5m':Dataset_Custom,
            'JFNG_data':Dataset_Custom,
            'JFNG_data_15min':Dataset_Custom,
            'JFNG_data_5min':Dataset_Custom,
            'JFNG_data_15min_unrh':Dataset_Custom,
            'JFNG_data_15min_unsp':Dataset_Custom,
            'JFNG_data_15min_unt2m':Dataset_Custom,
            'JFNG_data_15min_unwind':Dataset_Custom,
            'JFNG_data_15min_handle':Dataset_Custom,
            'JFNG_data_15min_1year':Dataset_Custom,
            'JFNG_data_1h': Dataset_Custom,
        }


        # 增加代码
        if args.data not in data_dict.keys():
            data_dict.update({args.data:"Dataset_Custom"})



        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag == 'tune':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            Data = Dataset_Finetune
        elif flag == 'tune_test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            Data = Dataset_Finetune
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Finetune_pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        #criterion =  nn.MSELoss()
        criterion = nn.MSELoss(reduce=False)
        # weighted MSE: no mean operation, return every element of minibatch
        return criterion

    def vali(self, vali_data, vali_loader, criterion,itr=None,plot_flag=0):
        self.model.eval()
        total_loss = []
        preds=[]
        trues=[]
        # 230809: TS score
        tp=0
        tn=0
        fp=0
        fn=0
        th1=0.5
        th2=0.5
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,class_label) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            loss_type = 'mse'
            if (loss_type == 'mse'):
                loss = criterion(pred.detach().cpu(),
                                 true.detach().cpu())  # to cpu due to no need to update parameter in validation
                loss = loss.mean(dim=1)
                loss = loss.mean(dim=1)
                # loss = loss.mean()

                # # weighted MSE loss
                weight = class_label.to(torch.device('cpu'))
                weighted_loss = weight * loss
                loss = weighted_loss.mean()
            if (loss_type == 'dilate'):
                alpha = 0.5
                gamma = 0.001
                size_pred = pred.shape
                for loss_i in range(size_pred[0]):
                    tmp_loss, loss_shape, loss_temporal = dilate_loss(
                        pred[loss_i].reshape(1, size_pred[1], size_pred[2]).detach().cpu(),
                        true[loss_i].reshape(1, size_pred[1], size_pred[2]).detach().cpu(), alpha, gamma, torch.device('cpu'))
                    tmp_loss = tmp_loss.unsqueeze(0)
                    if loss_i == 0:
                        loss = tmp_loss
                    else:
                        loss = torch.cat((loss, tmp_loss))
                # weighted DILATE loss
                weight = class_label.to(torch.device('cpu'))
                # weight=torch.ones(size_pred[0]).to(self.device)
                weighted_loss = weight * loss
                loss = weighted_loss.mean()

            total_loss.append(loss)
            temp_score=Score(pred.detach().cpu(), true.detach().cpu(), th1, th2)
            tp = temp_score[0] + tp
            fp = temp_score[1] + fp
            fn = temp_score[2] + fn
            tn = temp_score[3] + tn

        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        pred_plt = preds[::trues.shape[-2], :, :]
        pred_plt = pred_plt.reshape(-1, 1)
        true_plt = trues[::trues.shape[-2], :, :]
        true_plt = true_plt.reshape(-1, 1)
        fig,ax = plt.subplots(2,1,sharey='all', figsize=(18, 6),dpi=100)
        plt.subplot(211)
        plt.plot(pred_plt)
        plt.ylim(0,15)
        plt.subplot(212)
        plt.plot(true_plt)
        plt.plot(pred_plt)

        # plt.show()
        # self.writer.
        if plot_flag == 1:
            if itr is not None:
                self.writer.add_figure(tag='test prediction results', figure=fig,global_step=itr)
            else:
                self.writer.add_figure(tag='test prediction results', figure=fig)

        total_loss = np.average(total_loss)
        ts =  tp / (tp + fp + fn)
        far = 0 if (tp + fp)==0 else fp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        pod = tp / (tp + fn)
        hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        score=[acc,pod,far,ts,hss]
        self.model.train()
        return total_loss,score

    def train(self, setting):
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",

            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10,
            }
        )
        log_name=''.join(['runs/',setting,'/'])
        self.writer = SummaryWriter(log_dir=log_name, comment='test_tensorboard')

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,class_label) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    
                loss_type = 'mse'
                if (loss_type == 'mse'):
                    loss = criterion(pred, true)
                    loss = loss.mean(dim=1)
                    loss = loss.mean(dim=1)
                    # weighted MSE loss
                    weight = class_label.to(self.device)
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()
                if (loss_type == 'dilate'):
                    alpha = 0.5
                    gamma = 0.001
                    size_pred = pred.shape
                    for loss_i in range(size_pred[0]):
                        tmp_loss, loss_shape, loss_temporal = dilate_loss(pred[loss_i].reshape(1,size_pred[1],size_pred[2]), true[loss_i].reshape(1,size_pred[1],size_pred[2]), alpha, gamma, self.device)
                        tmp_loss = tmp_loss.unsqueeze(0)
                        if loss_i == 0:
                            loss = tmp_loss
                        else:
                            loss = torch.cat((loss, tmp_loss))
                    # weighted DILATE loss
                    weight = class_label.to(self.device)
                    #weight=torch.ones(size_pred[0]).to(self.device)
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()
                
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss,vali_score = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_score = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            tags = ["train_loss", "vali_loss", "test_loss", "learning_rate","Accuracy","POD","FAR","TS"]
            self.writer.add_scalar(tags[0], train_loss, epoch)
            self.writer.add_scalar(tags[1], vali_loss, epoch)
            self.writer.add_scalar(tags[2], test_loss, epoch)
            self.writer.add_scalar(tags[3], model_optim.param_groups[0]['lr'], epoch)
            self.writer.add_scalar(tags[4], test_score[0], epoch)
            self.writer.add_scalar(tags[5], test_score[1], epoch)
            self.writer.add_scalar(tags[6], test_score[2], epoch)
            self.writer.add_scalar(tags[7], test_score[3], epoch)

            # log metrics to wandb
            wandb.log({"test_loss": test_loss, "train_loss": train_loss})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        wandb.finish()
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []
        # add code 7/8
        batch_xs = []



        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,class_label) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            # add code 7/8
            batch_xs.append(batch_x.detach().cpu().numpy())


        preds = np.array(preds)
        trues = np.array(trues)
        # add code 7/8
        batch_xs = np.array(batch_xs)

        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        pred_plt = preds[::trues.shape[-2], :, :]
        pred_plt = pred_plt.reshape(-1, 1)
        true_plt = trues[::trues.shape[-2], :, :]
        true_plt = true_plt.reshape(-1, 1)
        fig = plt.figure(1, figsize=(8, 6))
        plt.subplot(211)
        plt.plot(pred_plt)
        plt.subplot(212)
        plt.plot(true_plt)
        plt.show()
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)


        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        # add code 7/8
        # np.save(folder_path+'batch_x.npy', batch_xs)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        now1 = datetime.datetime.now()
        np.save(folder_path+'real_prediction_{}.npy'.format(now1), preds)
        
        return

    #23.08.28 ywj: fine tune
    def finetuen(self, setting,load=True):
        # tensorboard
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        log_name=''.join(['runs_FT/',setting,'/',TIMESTAMP,'/'])
        self.writer = SummaryWriter(log_dir=log_name, comment='test_tensorboard')
        
        tune_data, tune_loader = self._get_data(flag='tune') # get small sample data
        tune_test_data, tune_test_loader = self._get_data(flag='tune_test') # get fine tune test dataset

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # output =self.model.projection.weight.data
        path = path + '/' + 'finetune' # set finetune path
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(tune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # small learning rate for model tuning
        self.args.learning_rate=0.00001
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # initial model results
        train_loss, train_score = self.vali(tune_data, tune_loader, criterion,0,plot_flag=1)
        test_loss, test_score = self.vali(tune_test_data, tune_test_loader, criterion,0,plot_flag=1)



        for epoch in range(self.args.train_epochs):
            iter_count = 0
            tune_loss = []
            preds = []
            trues = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, class_label) in enumerate(tune_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    tune_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

                loss_type = 'mse'
                if (loss_type == 'mse'):
                    loss = criterion(pred, true)
                    loss = loss.mean(dim=1)
                    loss = loss.mean(dim=1)
                    # weighted MSE loss
                    weight = class_label.to(self.device)
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()
                if (loss_type == 'dilate'):
                    alpha = 0.5
                    gamma = 0.001
                    size_pred = pred.shape
                    for loss_i in range(size_pred[0]):
                        tmp_loss, loss_shape, loss_temporal = dilate_loss(
                            pred[loss_i].reshape(1, size_pred[1], size_pred[2]),
                            true[loss_i].reshape(1, size_pred[1], size_pred[2]), alpha, gamma, self.device)
                        tmp_loss = tmp_loss.unsqueeze(0)
                        if loss_i == 0:
                            loss = tmp_loss
                        else:
                            loss = torch.cat((loss, tmp_loss))
                    # weighted DILATE loss
                    weight = class_label.to(self.device)
                    # weight=torch.ones(size_pred[0]).to(self.device)
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()

                tune_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            preds = np.array(preds)
            trues = np.array(trues)

            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            pred_plt=preds[::trues.shape[-2],:,:]
            pred_plt=pred_plt.reshape(-1,1)
            true_plt=trues[::trues.shape[-2],:,:]
            true_plt=true_plt.reshape(-1,1)
            fig, ax = plt.subplots(2, 1, sharey='all', figsize=(18, 6), dpi=100)
            plt.subplot(211)
            plt.plot(pred_plt)
            plt.subplot(212)
            plt.plot(true_plt)
            self.writer.add_figure(tag='train prediction results',figure=fig,global_step=epoch)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            tune_loss = np.average(tune_loss)
            test_loss, test_score = self.vali(tune_test_data, tune_test_loader, criterion,epoch+1,plot_flag=1)

            print("Epoch: {0}, Steps: {1} | Fine Tune Train Loss: {2:.7f} Fine Tune Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, tune_loss, test_loss))
            tags = ["train_loss", "vali_loss", "test_loss", "learning_rate", "Accuracy", "POD", "FAR", "TS"]
            self.writer.add_scalar(tags[0], tune_loss, epoch)
            self.writer.add_scalar(tags[2], test_loss, epoch)
            self.writer.add_scalar(tags[3], model_optim.param_groups[0]['lr'], epoch)
            self.writer.add_scalar(tags[4], test_score[0], epoch)
            self.writer.add_scalar(tags[5], test_score[1], epoch)
            self.writer.add_scalar(tags[6], test_score[2], epoch)
            self.writer.add_scalar(tags[7], test_score[3], epoch)
            # log metrics to wandb
            # wandb.log({"test_loss": test_loss, "train_loss": train_loss})

            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # wandb.finish()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def tune_test(self, setting):
        tune_test_data, tune_test_loader = self._get_data(flag='tune_test') # get fine tune test dataset
        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            total_loss=[]
            # 230809: TS score
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            th1 = 0.5
            th2 = 0.5
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, class_label) in enumerate(tune_test_loader):
                pred, true = self._process_one_batch(
                    tune_test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                loss_type = 'mse'
                if (loss_type == 'mse'):
                    loss = criterion(pred.detach().cpu(),
                                     true.detach().cpu())  # to cpu due to no need to update parameter in validation
                    loss = loss.mean(dim=1)
                    loss = loss.mean(dim=1)
                    # loss = loss.mean()

                    # # weighted MSE loss
                    weight = class_label.to(torch.device('cpu'))
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()
                if (loss_type == 'dilate'):
                    alpha = 0.5
                    gamma = 0.001
                    size_pred = pred.shape
                    for loss_i in range(size_pred[0]):
                        tmp_loss, loss_shape, loss_temporal = dilate_loss(
                            pred[loss_i].reshape(1, size_pred[1], size_pred[2]).detach().cpu(),
                            true[loss_i].reshape(1, size_pred[1], size_pred[2]).detach().cpu(), alpha, gamma,
                            torch.device('cpu'))
                        tmp_loss = tmp_loss.unsqueeze(0)
                        if loss_i == 0:
                            loss = tmp_loss
                        else:
                            loss = torch.cat((loss, tmp_loss))
                    # weighted DILATE loss
                    weight = class_label.to(torch.device('cpu'))
                    # weight=torch.ones(size_pred[0]).to(self.device)
                    weighted_loss = weight * loss
                    loss = weighted_loss.mean()

                total_loss.append(loss)
                temp_score = Score(pred.detach().cpu(), true.detach().cpu(), th1, th2)
                tp = temp_score[0] + tp
                fp = temp_score[1] + fp
                fn = temp_score[2] + fn
                tn = temp_score[3] + tn

            preds = np.array(preds)
            trues = np.array(trues)

            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            pred_plt = preds[::trues.shape[-2], :, :]
            pred_plt = pred_plt.reshape(-1, 1)
            true_plt = trues[::trues.shape[-2], :, :]
            true_plt = true_plt.reshape(-1, 1)
            fig, ax = plt.subplots(2, 1, sharey='all', figsize=(18, 6), dpi=100)
            plt.subplot(211)
            plt.plot(pred_plt)
            plt.subplot(212)
            plt.plot(true_plt)
            self.writer.add_figure(tag='Final prediction results', figure=fig)

            total_loss = np.average(total_loss)
            ts = tp / (tp + fp + fn)
            far = 0 if (tp + fp) == 0 else fp / (tp + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            pod = tp / (tp + fn)
            hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
            test_score = [acc, pod, far, ts, hss]
            self.writer.add_hparams({'learning rate': self.args.learning_rate,
                                     'seq_len':self.args.seq_len,
                                     'pred_len':self.args.pred_len,
                                     'label_len': self.args.label_len,
                                     'd_model': self.args.d_model,
                                     'e_layers': self.args.e_layers,
                                     'd_layers': self.args.d_layers,
                                     'train_epochs':self.args.train_epochs,
                                     'batch_size':self.args.batch_size,
                                     'dropout':self.args.dropout,
                                     },{'hparm/pod':test_score[1]},run_name='./hp_params')
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)

            print('mse:{}, mae:{}'.format(mse, mae))

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
            # add code 7/8
            # np.save(folder_path+'batch_x.npy', batch_xs)

        return

    def tune_predict(self, setting, load=True):
        pred_data, pred_loader = self._get_data(flag='pred')
        criterion = self._select_criterion()
        finetune=1  # choose fintune model or original model
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            if finetune:
                path = path + '/' + 'finetune' # set finetune path
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,class_label) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)


        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        pred_plt = preds[::trues.shape[-2], :, :]
        pred_plt = pred_plt.reshape(-1, 1)
        true_plt = trues[::trues.shape[-2], :, :]
        true_plt = true_plt.reshape(-1, 1)
        # read date array from file
        df_raw = pd.read_csv(os.path.join(self.args.root_path,
                                          self.args.data_path))
        pred_idx = df_raw[df_raw['date'] >= '2023/8/01 0:00'].index[0]
        df_raw = df_raw.drop(df_raw.index[0:pred_idx])
        date_array=df_raw['date'].to_numpy()
        date_array = [datetime.strptime(d, '%Y/%m/%d %H:%M') for d in date_array]
        # plot the figure
        fig, ax = plt.subplots(2, 1, sharey='all', figsize=(18, 6), dpi=100)
        plt.subplot(211)
        plt.plot(date_array[0:pred_plt.shape[0]],pred_plt)
        plt.subplot(212)
        plt.plot(date_array[0:pred_plt.shape[0]],true_plt)
        plt.show(block=True)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        now1 = datetime.now()
        np.save(folder_path + 'real_prediction_{}.npy'.format(now1), preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y


