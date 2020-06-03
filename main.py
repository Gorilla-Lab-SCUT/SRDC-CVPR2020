#####################################################################################
#                                                                                   #
# All the codes about the model constructing should be kept in the folder ./models/ #
# All the codes about the data process should be kept in the folder ./data/         #
# The file ./opts.py stores the options                                             #
# The file ./trainer.py stores the training and test strategy                       #
# The ./main.py should be simple                                                    #
#                                                                                   #
#####################################################################################
import os
import json
import shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.model_construct import Model_Construct # for the model construction
from trainer import train # for the training process
from trainer import validate, validate_compute_cen # for the validation/test process
from trainer import k_means, spherical_k_means, kernel_k_means # for K-means clustering and its variants
from trainer import source_select # for source sample selection
from opts import opts # options for the project
from data.prepare_data import generate_dataloader # prepare the data and dataloader
from utils.consensus_loss import ConsensusLoss
import time
import ipdb
import gc

args = opts()

best_prec1 = 0
best_test_prec1 = 0
cond_best_test_prec1 = 0
best_cluster_acc = 0 
best_cluster_acc_2 = 0 

def main():
    global args, best_prec1, best_test_prec1, cond_best_test_prec1, best_cluster_acc, best_cluster_acc_2
    
    # define model
    model = Model_Construct(args)
    print(model)
    model = torch.nn.DataParallel(model).cuda() # define multiple GPUs
    
    # define learnable cluster centers
    learn_cen = Variable(torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0))
    learn_cen.requires_grad_(True)
    learn_cen_2 = Variable(torch.cuda.FloatTensor(args.num_classes, args.num_neurons * 4).fill_(0))
    learn_cen_2.requires_grad_(True)

    # define loss function/criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_cons = ConsensusLoss(nClass=args.num_classes, div=args.div).cuda()
    
    np.random.seed(1)  # may fix test data
    random.seed(1)
    torch.manual_seed(1)
    
    # apply different learning rates to different layer
    optimizer = torch.optim.SGD([
            {'params': model.module.conv1.parameters(), 'name': 'conv'},
            {'params': model.module.bn1.parameters(), 'name': 'conv'},
            {'params': model.module.layer1.parameters(), 'name': 'conv'},
            {'params': model.module.layer2.parameters(), 'name': 'conv'},
            {'params': model.module.layer3.parameters(), 'name': 'conv'},
            {'params': model.module.layer4.parameters(), 'name': 'conv'},
            {'params': model.module.fc1.parameters(), 'name': 'ca_cl'},
            {'params': model.module.fc2.parameters(), 'name': 'ca_cl'},
            {'params': learn_cen, 'name': 'conv'},
            {'params': learn_cen_2, 'name': 'conv'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=args.nesterov)
    
    # resume
    epoch = 0                                
    init_state_dict = model.state_dict()
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_test_prec1 = checkpoint['best_test_prec1']
            cond_best_test_prec1 = checkpoint['cond_best_test_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            learn_cen = checkpoint['learn_cen']
            learn_cen_2 = checkpoint['learn_cen_2']
            print("==> loaded checkpoint '{}'(epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from does not exist!', args.resume)
    
    # make log directory
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    cudnn.benchmark = True
    
    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_target, val_loader_target_t, val_loader_source = generate_dataloader(args)
    train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1))) # avoid using ground truth labels of target

    print('begin training')
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    num_itern_total = args.epochs * batch_number

    new_epoch_flag = False # if new epoch, new_epoch_flag=True
    test_flag = False # if test, test_flag=True
    
    src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1) # initialize source weights
    
    count_itern_each_epoch = 0
    for itern in range(epoch * batch_number, num_itern_total):
        # evaluate on the target training and test data
        if (itern == 0) or (count_itern_each_epoch == batch_number):
            prec1, c_s, c_s_2, c_t, c_t_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args)
            test_acc = validate(val_loader_target_t, model, criterion, epoch, args)
            test_flag = True
            
            # K-means clustering or its variants
            if ((itern == 0) and args.src_cen_first) or (args.initial_cluster == 2):
                cen = c_s
                cen_2 = c_s_2
            else:
                cen = c_t
                cen_2 = c_t_2
            if (itern != 0) and (args.initial_cluster != 0) and (args.cluster_method == 'kernel_kmeans'):
                cluster_acc, c_t = kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_cluster_acc)
                cluster_acc_2, c_t_2 = kernel_k_means(target_features_2, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_cluster_acc_2, change_target=False)
            elif args.cluster_method != 'spherical_kmeans':
                cluster_acc, c_t = k_means(target_features, target_targets, train_loader_target, epoch, model, cen, args, best_cluster_acc)
                cluster_acc_2, c_t_2 = k_means(target_features_2, target_targets, train_loader_target, epoch, model, cen_2, args, best_cluster_acc_2, change_target=False)
            elif args.cluster_method == 'spherical_kmeans':
                cluster_acc, c_t = spherical_k_means(target_features, target_targets, train_loader_target, epoch, model, cen, args, best_cluster_acc)
                cluster_acc_2, c_t_2 = spherical_k_means(target_features_2, target_targets, train_loader_target, epoch, model, cen_2, args, best_cluster_acc_2, change_target=False)
            
            # record the best accuracy of K-means clustering
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            if cluster_acc != best_cluster_acc:
                best_cluster_acc = cluster_acc
                log.write('\n                                                          best_cluster acc: %3f' % best_cluster_acc)
            if cluster_acc_2 != best_cluster_acc_2:
                best_cluster_acc_2 = cluster_acc_2
                log.write('\n                                                          best_cluster_2 acc: %3f' % best_cluster_acc_2)
            log.close()
            
            # re-initialize learnable cluster centers
            if args.init_cen_on_st:
                cen = (c_t + c_s) / 2# or c_srctar
                cen_2 = (c_t_2 + c_s_2) / 2# or c_srctar_2
            else:
                cen = c_t
                cen_2 = c_t_2
            #if itern == 0:
            learn_cen.data = cen.data.clone()
            learn_cen_2.data = cen_2.data.clone()
            
            # select source samples
            if (itern != 0) and (args.src_soft_select or args.src_hard_select):
                src_cs = source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, c_t.data.clone(), args)
            
            # use source pre-trained model to extract features for first clustering
            if (itern == 0) and args.src_pretr_first: 
                model.load_state_dict(init_state_dict)
                
            if itern != 0:
                count_itern_each_epoch = 0
                epoch += 1
            batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
            train_loader_target_batch = enumerate(train_loader_target)
            train_loader_source_batch = enumerate(train_loader_source)
            
            new_epoch_flag = True
            
            del source_features
            del source_features_2
            del source_targets
            del target_features
            del target_features_2
            del target_targets
            del pseudo_labels
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        elif (args.src.find('visda') != -1) and (itern % int(num_itern_total / 200) == 0):
            prec1, _, _, _, _, _, _, _, _, _, _, _, _, _ = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=False)
            test_acc = validate(val_loader_target_t, model, criterion, epoch, args)
            test_flag = True
        if test_flag:
            # record the best prec1 and save checkpoint
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            if prec1 > best_prec1:
                best_prec1 = prec1
                cond_best_test_prec1 = 0
                log.write('\n                                                                                 best val acc till now: %3f' % best_prec1)
            if test_acc > best_test_prec1:
                best_test_prec1 = test_acc
                log.write('\n                                                                                 best test acc till now: %3f' % best_test_prec1)
            ipdb.set_trace()
            is_cond_best = ((prec1 == best_prec1) and (test_acc > cond_best_test_prec1))
            if is_cond_best:
                cond_best_test_prec1 = test_acc
                log.write('\n                                                                                 cond best test acc till now: %3f' % cond_best_test_prec1)
            log.close()
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'learn_cen': learn_cen,
                'learn_cen_2': learn_cen_2,
                'best_prec1': best_prec1,
                'best_test_prec1': best_test_prec1,
                'cond_best_test_prec1': cond_best_test_prec1,
            }, is_cond_best, args)
            
            test_flag = False
        
        # early stop
        if epoch > args.stop_epoch:
                break

        # train for one iteration
        train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, learn_cen, learn_cen_2, criterion_cons, optimizer, itern, epoch, new_epoch_flag, src_cs, args)

        model = model.cuda()
        new_epoch_flag = False
        count_itern_each_epoch += 1
    
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n***   best val acc: %3f   ***' % best_prec1)
    log.write('\n***   best test acc: %3f   ***' % best_test_prec1)
    log.write('\n***   cond best test acc: %3f   ***' % cond_best_test_prec1)
    # end time
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    if args.src_cls:
        batch_number_s = len(train_loader_source)
        if batch_number_s > batch_number_t:
            batch_number = batch_number_s
    
    return batch_number
    

def save_checkpoint(state, is_best, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()


