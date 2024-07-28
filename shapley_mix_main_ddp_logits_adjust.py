#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import torch.distributed as dist
from shapley_value import weight_calcu, weight_calcu_shapley
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import torch.nn.functional as F
from skelemix import *
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

#from torchlight.torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--dataset', default='ntu', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--naive-mix-epoch',
        type=int,
        default=5,
        help='epoch for the initial part estimation with naive mix')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=24,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        #action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        #action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        #action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--dist-weights',
        default=None,
        help='weights of part distribution')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--ema', type=float, default=0.9, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--naive-estimate-epoch',
        type=int,
        default=5,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.part_distri = torch.zeros(self.arg.model_args['num_class'], 10+10).cuda() + 1e-6#label_num, part_combination_num C(5,2) + C(5,3)
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name) and self.arg.local_rank<=0:
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                if self.arg.local_rank<=0:
                    self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                    self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                if self.arg.local_rank<=0:
                    self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_data()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.cuda(self.output_device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.output_device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.arg.local_rank], output_device=self.arg.local_rank)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        Feeder_test = import_class('feeders.feeder_ntu.Feeder_LT_nosample')
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_d = Feeder(**self.arg.train_feeder_args)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_d)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_d,
                batch_size=self.arg.batch_size//torch.distributed.get_world_size(),
                #shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                sampler=train_sampler,
                worker_init_fn=init_seed)
        self.cls_num_list = self.data_loader['train'].dataset.num_per_cls_dict
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder_test(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        self.output_device = self.arg.local_rank#output_device
        output_device = self.output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        #print(Model)
        self.model = Model(**self.arg.model_args)
        #print(self.model)
        self.loss = LogitAdjust(self.cls_num_list).cuda(output_device)

        if self.arg.weights:
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights,map_location='cpu')

            dist = np.load(self.arg.dist_weights)
            self.part_distri = torch.from_numpy(dist).cuda()
            print('Load part distribution from ', self.arg.dist_weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir) and self.arg.local_rank<=0:
            os.makedirs(self.arg.work_dir)
        if self.arg.local_rank<=0:
            with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
                f.write(f"# command line: {' '.join(sys.argv)}\n\n")
                yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if self.arg.local_rank>0:
            return
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def map_list2label(self, list):
        #01 02 03 04 12 13 14 23 24 34
        mapping = {
            '01': 0, '02': 1, '03': 2, '04': 3, '12': 4,
            '13': 5, '14': 6, '23': 7, '24': 8, '34': 9,
            '012': 10, '013': 11, '014': 12, '023': 13, '024': 14,
            '034': 15, '123': 16, '124': 17, '134': 18, '234': 19,
        }
        list.sort()
        a = ''
        for i in list:
            a+=str(i)
        return mapping[a]

    @torch.no_grad()
    def update_part_dist(self, epoch, iter, spa_label, weight, label):
        warm_epoch = self.arg.naive_estimate_epoch
        if epoch > warm_epoch:
            ema_decay = self.arg.ema
        else:
            ema_decay = 0.0
        # update weight
        for i in range(len(label)):
            self.part_distri[label[i]][spa_label] = self.part_distri[label[i]][spa_label] * ema_decay + weight[i] * (1 - ema_decay)

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        print('device', self.arg.device)
        loss_value = []
        loss_value2 = []
        acc_value = []
        if self.arg.local_rank<=0:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        loader.sampler.set_epoch(epoch)
        process = tqdm(loader, ncols=40)
        #NOTE if not long tail, ignore the following code
        label_dist = self.arg.model_args['num_class'] - np.array(list(range(0,self.arg.model_args['num_class'])))
        label_dist = torch.from_numpy(label_dist).cuda()# the lower the training sample, the larger priority in the label dist
        
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            ################ part dist estimation ####################
            estimation_interval = 1 #rfequency of performing estimation 
            if epoch < self.arg.naive_mix_epoch or batch_idx % estimation_interval == 0:
                Cs = random.randint(2, 3)
                spa_list = random.sample(list(range(5)), Cs)
                spa_label = self.map_list2label(spa_list)
                if epoch < self.arg.naive_estimate_epoch:
                    weight = weight_calcu(model=self.model, data=data, label=label, spa_mask_list=spa_list, average_motion=loader.dataset.avg_motion).cuda()
                else:
                    weight = weight_calcu_shapley(model=self.model, data=data, label=label, spa_mask_list=spa_list, average_motion=loader.dataset.avg_motion).cuda()
                
                part_each_num = np.array([5,6,6,4,4])#the joint num of each parts
                weight = weight/part_each_num[spa_list].sum()*10 # node normlization (*10 just for print debug)
                self.update_part_dist(epoch, self.global_step, spa_label, weight, label)
            ##########################################################

            # mix generation
            if epoch < self.arg.naive_mix_epoch:
                randidx, mix_data, mask = ske_swap_randscale(data,2, 3, 7, 11)
                lamb = mask.mean()
            else:
                dist.all_reduce(self.part_distri, op=dist.ReduceOp.SUM)
                self.part_distri = self.part_distri / torch.distributed.get_world_size()
                part_dist = self.part_distri[label] 
                randidx, mix_data, lamb = ske_swap_randscale_sample_noweighted(data,2, 3, 7, 11, part_dist, label_dist[label])#, adatemp)
                

            output = self.model(torch.cat([data,mix_data],dim=0))
            output, mix_output = torch.chunk(output,2,dim=0)

            label_onehot = torch.zeros((data.shape[0], output.shape[1])).cuda()
            label_onehot.scatter_(1,label.unsqueeze(-1),1)
            label_mix = label_onehot * (1 - lamb) + label_onehot[randidx] * lamb

            loss1 = self.loss(output, label)
            mix_output = F.log_softmax(mix_output+self.loss.m_list, dim=1)
            loss2 = -torch.mean(torch.sum(mix_output * label_mix, dim=1))

            loss = loss1 + loss2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss1.data.item())
            loss_value2.append(loss2.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            if self.arg.local_rank<=0:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.arg.local_rank<=0:
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}. Mean mix training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(loss_value2), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
        
        if save_model and self.arg.local_rank<=0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(self.current_acc) + '.pt')
            np.save(self.arg.dataset+'part_dist_'+str(epoch+1)+'.npy', self.part_distri.detach().cpu().numpy())

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None and self.arg.local_rank<=0:
            f_w = open(wrong_file, 'w')
        if result_file is not None and self.arg.local_rank<=0:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            self.current_acc = accuracy
            best = False
            if accuracy > self.best_acc:
                best = True
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
            
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name, 'Best Accuracy: ', self.best_acc)
            self.print_log(f'Accuracy: {accuracy} Best Accuracy: {self.best_acc}')
            if self.arg.phase == 'train' and self.arg.local_rank<=0:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if best and self.arg.local_rank<=0:
                with open('{}/best_score.pkl'.format(
                        self.arg.work_dir), 'wb+') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(pred_list, label_list, self.data_loader['train'],
                                                                acc_per_cls=False)
            print('Many Accuracy: ', many_acc_top1, 'Median Accuracy: ', median_acc_top1, 'Few Accuracy: ', low_acc_top1)
            self.print_log(f'Many Accuracy:, {many_acc_top1}, Median Accuracy:, {median_acc_top1}, Few Accuracy: , {low_acc_top1}')
            if self.arg.local_rank<=0:
                with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(each_acc)
                    writer.writerows(confusion)
            dist.barrier()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                if epoch == self.arg.naive_mix_epoch-1 or epoch == 59:
                    save_model=True
                self.train(epoch, save_model=save_model)

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.label).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
    
if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    parser.add_argument("--local_rank", default=-1, type=int)
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    local_rank = arg.local_rank
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
