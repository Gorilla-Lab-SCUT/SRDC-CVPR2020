import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.kernel_kmeans import KernelKMeans
import gc
import ipdb

def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, learn_cen, learn_cen_2, criterion_cons, optimizer, itern, epoch, new_epoch_flag, src_cs, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_source = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    model.train()

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty parameter
    #lam = 1.0
    if args.src_cls:
        weight = lam
    else:
        weight = 1.0
    adjust_learning_rate(optimizer, epoch, args) # adjust learning rate

    end = time.time()
    # prepare target data
    try:
        if args.aug_tar_agree and (not args.gray_tar_agree):
            (input_target, input_target_dup, target_target, _) = train_loader_target_batch.__next__()[1]
        elif args.gray_tar_agree and (not args.aug_tar_agree):
            (input_target, input_target_gray, target_target, _) = train_loader_target_batch.__next__()[1]
        elif args.aug_tar_agree and args.gray_tar_agree:
            (input_target, input_target_dup, input_target_gray, target_target, _) = train_loader_target_batch.__next__()[1]
        else:
            (input_target, target_target, _) = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        if args.aug_tar_agree and (not args.gray_tar_agree):
            (input_target, input_target_dup, target_target, _) = train_loader_target_batch.__next__()[1]
        elif args.gray_tar_agree and (not args.aug_tar_agree):
            (input_target, input_target_gray, target_target, _) = train_loader_target_batch.__next__()[1]
        elif args.aug_tar_agree and args.gray_tar_agree:
            (input_target, input_target_dup, input_target_gray, target_target, _) = train_loader_target_batch.__next__()[1]
        else:
            (input_target, target_target, _) = train_loader_target_batch.__next__()[1]
    target_target = target_target.cuda(async=True)
    input_target_var = Variable(input_target)
    target_target_var = Variable(target_target)
    if args.aug_tar_agree:
        input_target_dup_var = Variable(input_target_dup)
    if args.gray_tar_agree:
        input_target_gray_var = Variable(input_target_gray)
    
    # model forward on target
    f_t, f_t_2, ca_t = model(input_target_var)
    if args.aug_tar_agree:
        _, _, ca_t_dup = model(input_target_dup_var)
    if args.gray_tar_agree:
        _, _, ca_t_gray = model(input_target_gray_var)
    
    loss = 0
    if args.aug_tar_agree and (not args.gray_tar_agree):
        loss += weight * criterion_cons(ca_t, ca_t_dup)
    elif args.gray_tar_agree and (not args.aug_tar_agree):
        loss += weight * criterion_cons(ca_t, ca_t_gray)
    elif args.aug_tar_agree and args.gray_tar_agree:
        loss += weight * (criterion_cons(ca_t, ca_t_dup) + criterion_cons(ca_t, ca_t_gray))
                
    loss += weight * TarDisClusterLoss(args, epoch, ca_t, target_target, em=(args.cluster_method == 'em'))
    
    if args.learn_embed:
        prob_pred = (1 + (f_t.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
        loss += weight * TarDisClusterLoss(args, epoch, prob_pred, target_target, softmax=args.embed_softmax)
        if not args.no_second_embed:
            prob_pred_2 = (1 + (f_t_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
            loss += weight * TarDisClusterLoss(args, epoch, prob_pred_2, target_target, softmax=args.embed_softmax)
        
    if args.src_cls:
        # prepare source data
        try:
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
        except StopIteration:
            train_loader_source_batch = enumerate(train_loader_source)
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
        target_source = target_source.cuda(async=True)
        input_source_var = Variable(input_source)
        target_source_var = Variable(target_source)
        
        # model forward on source
        f_s, f_s_2, ca_s = model(input_source_var)
        prec1_s = accuracy(ca_s.data, target_source, topk=(1,))[0]
        top1_source.update(prec1_s.item(), input_source.size(0))
        
        loss += SrcClassifyLoss(args, ca_s, target_source, index, src_cs, lam, fit=args.src_fit)
        
        if args.learn_embed:
            prob_pred = (1 + (f_s.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
            loss += weight * SrcClassifyLoss(args, prob_pred, target_source, index, src_cs, lam, softmax=args.embed_softmax, fit=args.src_fit)
            if not args.no_second_embed:
                prob_pred_2 = (1 + (f_s_2.unsqueeze(1) - learn_cen_2.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
                loss += weight * SrcClassifyLoss(args, prob_pred_2, target_source, index, src_cs, lam, softmax=args.embed_softmax, fit=args.src_fit)

    losses.update(loss.data.item(), input_target.size(0))
    
    # loss backward and network update
    model.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    if itern % args.print_freq == 0:
        print('Train - epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'S@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               epoch, args.epochs, batch_time=batch_time,
               data_time=data_time, s_top1=top1_source, loss=losses))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\nTrain - epoch: %d, top1_s acc: %3f, loss: %4f" % (epoch, top1_source.avg, losses.avg))
        log.close()
    if new_epoch_flag:
        print('The penalty weight is %3f' % weight)
    
    return train_loader_source_batch, train_loader_target_batch


def TarDisClusterLoss(args, epoch, output, target, softmax=True, em=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda()) # assigned pseudo labels
        if (epoch == 0) or args.ao:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2
    
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()
    
    return loss
    
    
def SrcClassifyLoss(args, output, target, index, src_cs, lam, softmax=True, fit=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    if fit:
        prob_q = (1 - prob_p) * prob_q + prob_p * prob_p    
    if args.src_mix_weight:
        src_weights = lam * src_cs[index] + (1 - lam) * torch.ones(output.size(0)).cuda()
    else:
        src_weights = src_cs[index]
    
    if softmax:
        loss = - (src_weights * (prob_q * F.log_softmax(output, dim=1)).sum(1)).mean()
    else:
        loss = - (src_weights * (prob_q * prob_p.log()).sum(1)).mean()
    
    return loss


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    total_vector = torch.FloatTensor(args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0)
    
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # forward
        with torch.no_grad():
            _, _, output = model(input_var)
            loss = criterion(output, target_var)

        # compute and record loss and accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector, correct_vector) # compute class-wise accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test on T test set - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, 
                   loss=losses, top1=top1, top5=top5))

    acc_for_each_class = 100.0 * correct_vector / total_vector
    print(' * Test on T test set - Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Test on T test set - epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))
    if args.src.find('visda') != -1:
        log.write("\nAcc for each class: ")
        for i in range(args.num_classes):
            if i == 0:
                log.write("%dst: %3f" % (i+1, acc_for_each_class[i]))
            elif i == 1:
                log.write(",  %dnd: %3f" % (i+1, acc_for_each_class[i]))
            elif i == 2:
                log.write(", %drd: %3f" % (i+1, acc_for_each_class[i]))
            else:
                log.write(", %dth: %3f" % (i+1, acc_for_each_class[i]))
        log.write("\n                          Avg. over all classes: %3f" % acc_for_each_class.mean())
        log.close()
        return acc_for_each_class.mean()
    else:
        log.close()
        return top1.avg

    
def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    source_features = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), 2048).fill_(0)
    source_features_2 = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), args.num_neurons*4).fill_(0)
    source_targets = torch.cuda.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0)
    c_src = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_src_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
    count_s = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    if compute_cen:
        for i, (input, target, index) in enumerate(val_loader_source): # the iterarion in the source dataset
            input_var = Variable(input)
            target = target.cuda(async=True)
            with torch.no_grad():
                feature, feature_2, output = model(input_var)
            source_features[index.cuda(), :] = feature.data.clone()
            source_features_2[index.cuda(), :] = feature_2.data.clone()
            source_targets[index.cuda()] = target.clone()
            target_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            target_.scatter_(1, target.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            if args.cluster_method == 'spherical_kmeans':
                c_src += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * target_.unsqueeze(2)).sum(0)
            else:
                c_src += (feature.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                c_src_2 += (feature_2.unsqueeze(1) * target_.unsqueeze(2)).sum(0)
                count_s += target_.sum(0).unsqueeze(1)
    
    target_features = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), 2048).fill_(0)
    target_features_2 = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_neurons*4).fill_(0)
    target_targets = torch.cuda.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0)
    pseudo_labels = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), args.num_classes).fill_(0)    
    c_tar = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_tar_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
    count_t = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    
    total_vector = torch.FloatTensor(args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0)
    
    end = time.time()
    for i, (input, target, index) in enumerate(val_loader_target): # the iterarion in the target dataset
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)
        
        with torch.no_grad():
            feature, feature_2, output = model(input_var)
        
        target_features[index.cuda(), :] = feature.data.clone() # index:a tensor 
        target_features_2[index.cuda(), :] = feature_2.data.clone()
        target_targets[index.cuda()] = target.clone()
        pseudo_labels[index.cuda(), :] = output.data.clone()
            
        if compute_cen: # compute target class centroids
            pred = output.data.max(1)[1]
            pred_ = torch.cuda.FloatTensor(output.size()).fill_(0)
            pred_.scatter_(1, pred.unsqueeze(1), torch.ones(output.size(0), 1).cuda())
            if args.cluster_method == 'spherical_kmeans':
                c_tar += ((feature / feature.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                c_tar_2 += ((feature_2 / feature_2.norm(p=2, dim=1, keepdim=True)).unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
            else:
                c_tar += (feature.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                c_tar_2 += (feature_2.unsqueeze(1) * pred_.unsqueeze(2)).sum(0)
                count_t += pred_.sum(0).unsqueeze(1)
        
        # compute and record loss and accuracy
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        total_vector, correct_vector = accuracy_for_each_class(output.data, target, total_vector, correct_vector) # compute class-wise accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test on T training set - [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'T@1 {tc_top1.val:.3f} ({tc_top1.avg:.3f})\t'
                  'T@5 {tc_top5.val:.3f} ({tc_top5.avg:.3f})\t'
                  'L {tc_loss.val:.4f} ({tc_loss.avg:.4f})'.format(
                   epoch, i, len(val_loader_target), batch_time=batch_time,
                   data_time=data_time, tc_top1=top1, tc_top5=top5, tc_loss=losses))

    # compute global class centroids
    c_srctar = torch.cuda.FloatTensor(args.num_classes, 2048).fill_(0)
    c_srctar_2 = torch.cuda.FloatTensor(args.num_classes, args.num_neurons*4).fill_(0)
    if (args.cluster_method == 'spherical_kmeans'):
        c_srctar = c_src + c_tar
        c_srctar_2 = c_src_2 + c_tar_2
    else:
        c_srctar = (c_src + c_tar) / (count_s + count_t)
        c_srctar_2 = (c_src_2 + c_tar_2) / (count_s + count_t)
        c_src /= count_s
        c_src_2 /= count_s
        c_tar /= (count_t + args.eps)
        c_tar_2 /= (count_t + args.eps)
        
    acc_for_each_class = 100.0 * correct_vector / total_vector
    
    print(' * Test on T training set - Prec@1 {tc_top1.avg:.3f}, Prec@5 {tc_top5.avg:.3f}'.format(tc_top1=top1, tc_top5=top5))

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\nTest on T training set - epoch: %d, tc_loss: %4f, tc_Top1 acc: %3f, tc_Top5 acc: %3f" % (epoch, losses.avg, top1.avg, top5.avg))
    
    if args.src.find('visda') != -1:
        log.write("\nAcc for each class: ")
        for i in range(args.num_classes):
            if i == 0:
                log.write("%dst: %3f" % (i+1, acc_for_each_class[i]))
            elif i == 1:
                log.write(",  %dnd: %3f" % (i+1, acc_for_each_class[i]))
            elif i == 2:
                log.write(", %drd: %3f" % (i+1, acc_for_each_class[i]))
            else:
                log.write(", %dth: %3f" % (i+1, acc_for_each_class[i]))
        log.write("\n                          Avg. over all classes: %3f" % acc_for_each_class.mean())
        log.close()
        
        return acc_for_each_class.mean(), c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels
    else:
        log.close()
        return top1.avg, c_src, c_src_2, c_tar, c_tar_2, c_srctar, c_srctar_2, source_features, source_features_2, source_targets, target_features, target_features_2, target_targets, pseudo_labels


def source_select(source_features, source_targets, target_features, pseudo_labels, train_loader_source, epoch, cen, args):
    # compute source weights
    source_cos_sim_temp = source_features.unsqueeze(1) * cen.unsqueeze(0)
    source_cos_sim = 0.5 * (1 + source_cos_sim_temp.sum(2) / (source_features.norm(2, dim=1, keepdim=True) * cen.norm(2, dim=1, keepdim=True).t() + args.eps))
    src_cs = torch.gather(source_cos_sim, 1, source_targets.unsqueeze(1)).squeeze(1)
    
    # or hard source sample selection
    if args.src_hard_select:
        num_select_src_each_class = torch.cuda.LongTensor(args.num_classes).fill_(0)
        tao = 1 / (1 + math.exp(- args.tao_param * (epoch + 1))) - 0.01
        delta = np.log(args.num_classes) / 10
        indexes = torch.arange(0, source_features.size(0))
        
        target_kernel_sim = (1 + (target_features.unsqueeze(1) - cen.unsqueeze(0)).pow(2).sum(2) / args.alpha).pow(- (args.alpha + 1) / 2)
        if args.embed_softmax:
            target_kernel_sim = F.softmax(target_kernel_sim, dim=1)
        else:
            target_kernel_sim /= target_kernel_sim.sum(1, keepdim=True)
        _, pseudo_cat_dist = target_kernel_sim.max(dim=1)
        pseudo_labels_softmax = F.softmax(pseudo_labels, dim=1)
        _, pseudo_cat_std = pseudo_labels_softmax.max(dim=1)
        
        selected_indexes = []
        for c in range(args.num_classes):
            _, idxes = src_cs[source_targets == c].sort(dim=0, descending=True)
            
            temp1 = target_kernel_sim[pseudo_cat_dist == c].mean(dim=0)
            temp2 = pseudo_labels_softmax[pseudo_cat_std == c].mean(dim=0)
            temp1 = - (temp1 * ((temp1 + args.eps).log())).sum(0) # entropy 1
            temp2 = - (temp2 * ((temp2 + args.eps).log())).sum(0) # entropy 2
            if (temp1 > delta) and (temp2 > delta):
                tao -= 0.1
            elif (temp1 <= delta) and (temp2 <= delta):
                pass
            else:
                tao -= 0.05
            while 1:
                num_select_src_each_class[c] = (src_cs[source_targets == c][idxes] >= tao).float().sum()
                if num_select_src_each_class[c] > 0: # at least 1
                    selected_indexes.extend(list(np.array(indexes[source_targets == c][idxes][src_cs[source_targets == c][idxes] >= tao])))
                    break
                else:
                    tao -= 0.05
        
        train_loader_source.dataset.samples = []
        train_loader_source.dataset.tgts = []
        for idx in selected_indexes:
            train_loader_source.dataset.samples.append(train_loader_source.dataset.imgs[idx])
            train_loader_source.dataset.tgts.append(train_loader_source.dataset.imgs[idx][1])
        print('%d source instances have been selected at %d epoch' % (len(selected_indexes), epoch))
        print('Number of selected source instances each class: ', num_select_src_each_class)
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\n~~~%d source instances have been selected at %d epoch~~~' % (len(selected_indexes), epoch))
        log.close()
        
        src_cs = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1)
    
    del source_cos_sim_temp
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return src_cs
    

def kernel_k_means(target_features, target_targets, pseudo_labels, train_loader_target, epoch, model, args, best_prec, change_target=True):
    # define kernel k-means clustering
    kkm = KernelKMeans(n_clusters=args.num_classes, max_iter=args.cluster_iter, random_state=0, kernel=args.cluster_kernel, gamma=args.gamma, verbose=1)
    kkm.fit(np.array(target_features.cpu()), initial_label=np.array(pseudo_labels.max(1)[1].long().cpu()), true_label=np.array(target_targets.cpu()), args=args, epoch=epoch)
    
    idx_sim = torch.from_numpy(kkm.labels_)
    c_tar = torch.cuda.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
    count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    for i in range(target_targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
        if change_target:
            train_loader_target.dataset.tgts[i] = idx_sim[i].item()
    c_tar /= (count + args.eps)
    
    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1
        #torch.save(c_tar, os.path.join(args.log, 'c_t_kernel_kmeans_cluster_best.pth.tar'))
        #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kernel_kmeans_cluster_best.pth.tar'))
    
    del target_features
    del target_targets
    del pseudo_labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar


def k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) - c_tar.unsqueeze(0)
        dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_kmeans_cluster_best.pth.tar'))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets, total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i+1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i+1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()
        
        c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
        count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0) 
        for k in range(args.num_classes):
            c_tar_temp[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()
        c_tar_temp /= (count + args.eps)
        
        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])
        
        c_tar = c_tar_temp.clone()
        
        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
    
    del target_features
    del target_targets
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar

    
def spherical_k_means(target_features, target_targets, train_loader_target, epoch, model, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.data.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - dist_xt_ct_temp.sum(2) / (target_features.norm(2, dim=1, keepdim=True) * c_tar.norm(2, dim=1, keepdim=True).t() + args.eps))
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
            #torch.save(c_tar, os.path.join(args.log, 'c_t_spherical_kmeans_cluster_best.pth.tar'))
            #torch.save(model.state_dict(), os.path.join(args.log, 'checkpoint_spherical_kmeans_cluster_best.pth.tar'))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, Spherical K-means clustering %d, Average clustering time %.3f, Prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
            total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets, total_vector_dist, correct_vector_dist)
            acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                if i == 0:
                    log.write("%dst: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i+1, acc_for_each_class_dist[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i+1, acc_for_each_class_dist[i]))
                else:
                    log.write(", %dth: %3f" % (i+1, acc_for_each_class_dist[i]))
            log.write("\n                          Avg_dist. over all classes: %3f" % acc_for_each_class_dist.mean())
        log.close()
        c_tar_temp = torch.cuda.FloatTensor(args.num_classes, c_tar.size(1)).fill_(0)
        for k in range(args.num_classes):
            c_tar_temp[k] += (target_features[idx_sim.squeeze(1) == k] / (target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + args.eps)).sum(0)
        
        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])
        
        c_tar = c_tar_temp.clone()
        
        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
    
    del target_features
    del target_targets
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    if args.lr_plan == 'step':
        exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
        lr = args.lr * (0.1 ** exp)
    elif args.lr_plan == 'dao':
        lr = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
    for param_group in optimizer.param_groups:
       if param_group['name'] == 'conv':
           param_group['lr'] = lr * 0.1
       elif param_group['name'] == 'ca_cl':
           param_group['lr'] = lr
       else:
           raise ValueError('The required parameter group does not exist.')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]
    
    return total_vector, correct_vector

