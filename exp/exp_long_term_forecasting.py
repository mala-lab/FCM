from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def my_kl_loss(self,p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def adjustment(self,gt, pred):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        pred = np.array(pred)
        gt = np.array(gt)

        return gt, pred
    
    def ad_metrics(self,gt, pred):
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        return accuracy,precision,recall,f_score
    
    def read_csv_result(self,real_label_path,pred_path):
        df_label = pd.read_csv(real_label_path)
        test_real_labels = df_label['labels']
        print(len(df_label))

        df = pd.read_csv(pred_path)
        test_energy = df['pred']
        test_labels = df['labels']
        print(len(df))
        
        differences = np.where(test_labels != test_real_labels)[0]
        print(len(differences))
        detec = np.where(test_real_labels == 1)[0]
        print(len(detec))

        # 一般阈值设置为99
        thred = np.percentile(test_energy, 99)
        pred_ = (test_energy > thred).astype(int)
        # print(np.where(pred_ == 1))
        gt_ = np.array(test_labels).astype(int)
        # print(np.where(gt_ == 1))

        return pred_,gt_,differences,detec
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,label,real_label) in enumerate(vali_loader):
                # print(i)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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
                        outputs,AD_enc_out, series, prior, sigmas = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.dataset == 'SWAT':
            module_epoch = 15000
        elif self.args.dataset == 'UCR':
            module_epoch = len(train_data)/self.args.batch_size - (len(train_data)/self.args.batch_size)/10
            print(module_epoch)
        elif self.args.dataset == 'SMD':
            module_epoch = 30000
        elif self.args.dataset == 'SMAP':
            module_epoch = 4000
        elif self.args.dataset == 'PSM':
            module_epoch = 5000
        elif self.args.dataset == 'MSL':
            module_epoch = 2500

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,test_labels,real_label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,AD_enc_out, series, prior, sigmas_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if i > module_epoch:
                        with torch.no_grad():
                            outputs_,AD_enc_out, series, prior, sigmas = self.model(outputs.detach(),batch_x_mark.detach(),batch_x.detach(),batch_y_mark.detach(),con_cat=True)


                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(self.my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.args.seq_len)).detach())) + torch.mean(
                            self.my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            self.args.seq_len)).detach(),
                                    series[u])))
                        prior_loss += (torch.mean(self.my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.args.seq_len)),
                            series[u].detach())) + torch.mean(
                            self.my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.args.seq_len)))))
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = criterion(AD_enc_out, batch_x)
                    
                    k = 3
                    # loss1_list.append((rec_loss - k * series_loss).item())
                    loss1 = rec_loss - k * series_loss
                    loss2 = rec_loss + k * prior_loss

                    # forecast loss
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                loss1.requires_grad_(True)
                loss2.requires_grad_(True)
                loss.requires_grad_(True)

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
                    loss1.backward(retain_graph=True)
                    loss2.backward(retain_graph=True)
                    loss.backward()
                    model_optim.step()


            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        test_labels = []
        real_labels = []
        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        epoch_time = time.time()

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,labels,real_label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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
                        outputs,AD_enc_out, series, prior, sigmas = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs_,AD_enc_out, series, prior, sigmas = self.model(outputs.detach(),batch_x_mark.detach(),batch_x.detach(),batch_y_mark.detach(),con_cat=True)
                    

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    temperature = 50
                    for u in range(len(prior)):
                        if u == 0:
                            series_loss = self.my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.args.seq_len)).detach()) * temperature
                            prior_loss = self.my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.args.seq_len)),
                                series[u].detach()) * temperature
                        else:
                            series_loss += self.my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.args.seq_len)).detach()) * temperature
                            prior_loss += self.my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.args.seq_len)),
                                series[u].detach()) * temperature
                    
                    criterion = nn.MSELoss(reduce=False)
                    rec_loss = torch.mean(criterion(batch_x,AD_enc_out), dim=-1)
                    metric_ = torch.softmax((-series_loss - prior_loss), dim=-1)

                    cri = metric_ * rec_loss
                    cri = cri.detach().cpu().numpy()
                    attens_energy.append(cri)
                    test_labels.append(labels)
                    real_labels.append(real_label)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            print("inference cost time: {}".format(time.time() - epoch_time))

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        real_labels = np.concatenate(real_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        real_labels = np.array(real_labels)

        import pandas as pd
        df = pd.DataFrame({'labels': real_labels.astype(int)})
        df.to_csv(folder_path+self.args.dataset+'_reallabel.csv', index=False)

        import pandas as pd
        df = pd.DataFrame({'labels': test_labels.astype(int),'pred':test_energy})
        df.to_csv(folder_path + self.args.dataset+'_result.csv', index=False)

        pred_,gt_,differences,detec = self.read_csv_result(folder_path+self.args.dataset+'_reallabel.csv',folder_path + self.args.dataset+'_result.csv')
        pred_ad = np.array(pred_[np.delete(np.arange(len(pred_)), detec)])
        gt_ad = np.array(gt_[np.delete(np.arange(len(gt_)), detec)])
        gt_ad, pred_ad = self.adjustment(gt_ad, pred_ad)
        accuracy,precision,recall,f_score = self.ad_metrics(gt_ad, pred_ad)

        # gt, pred = self.adjustment(gt_, pred_)
        # accuracy,precision,recall,f_score = self.ad_metrics(gt, pred)


        # # ad的结果
        # pred_for = np.array(pred_[np.delete(np.arange(len(pred_)), differences)])
        # gt_for = np.array(gt_[np.delete(np.arange(len(gt_)), differences)])
        # gt, pred = adjustment(gt_for, pred_for)
        # metrics(gt, pred)


        preds = np.array(preds)
        trues = np.array(trues)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},accuracy:{:0.4f},precision:{:0.4f},recall:{:0.4f},f_score:{:0.4f}'.format(mse, mae,accuracy,precision,recall,f_score))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # fig_start = differences[100]
        # fig_long = 200
        # fig_v = -10

        # gt = trues.reshape(-1,trues.shape[-1])[fig_start - fig_long:fig_start + fig_long, fig_v]
        # pd = preds.reshape(-1,trues.shape[-1])[fig_start - fig_long:fig_start + fig_long, fig_v]

        # gt_min = np.min(gt)
        # gt_max = np.max(gt)
        # # 使用归一化公式将数据归一化到0-10的范围
        # normalized_gt = 10 * (gt - gt_min) / (gt_max - gt_min)

        # pd_min = np.min(pd)
        # pd_max = np.max(pd)
        # # 使用归一化公式将数据归一化到0-10的范围
        # normalized_pd = 7 * (pd - pd_min) / (pd_max - pd_min)

        # visual(normalized_gt, normalized_pd, os.path.join(folder_path, 'fig.pdf'))
        # visual(gt, pd, os.path.join(folder_path, 'fig.pdf'))


        return
