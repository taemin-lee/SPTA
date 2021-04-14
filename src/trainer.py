import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, get_metric, AverageMeter
from src.datasets.ingredient import get_dataloader
from src.losses import SupConLoss
from src.optim import warmup_learning_rate

trainer_ingredient = Ingredient('trainer')
@trainer_ingredient.config
def config():
    print_freq = 10
    meta_val_way = 5
    meta_val_shot = 1
    meta_val_metric = 'cosine'  # ('euclidean', 'cosine', 'l1', l2')
    meta_val_iter = 500
    disable_train_augment = False
    meta_val_query = 15
    val_data_path = None  # Only for cross-domain scenario
    val_split_dir = None  # Only for cross-domain scenario
    meta_val_interval = 1
    alpha = - 1.0
    label_smoothing = 0.
    beta = - 1.0
    cutmix_prob = - 1.0

    temp = 0.1


class Trainer:
    @trainer_ingredient.capture
    def __init__(self, device, disable_train_augment, meta_val_iter, meta_val_way, meta_val_shot,
                 meta_val_query, val_data_path, val_split_dir, meta_val_interval, ex):

        self.train_loader = get_dataloader(split='train', aug=not disable_train_augment, shuffle=True, fixres=ex.current_run.config['fixres']).get()
        sample_info = [meta_val_iter, meta_val_way, meta_val_shot, meta_val_query]
        if val_data_path is not None:
            self.val_loader = get_dataloader('val', aug=False, path=val_data_path,
                                             split_dir=val_split_dir, sample=sample_info).get()
        else:
            self.val_loader = get_dataloader(split='val', aug=False, sample=sample_info).get()
        self.device = device
        self.meta_val_interval = meta_val_interval
        self.num_classes = ex.current_run.config['model']['num_classes']
        self.arch = ex.current_run.config['model']['arch']

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
    
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    
        return bbx1, bby1, bbx2, bby2

    @trainer_ingredient.capture
    def train(self, epoch, scheduler, print_freq, disable_tqdm, callback,
              model, alpha, beta, cutmix_prob, optimizer, fixres):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        if fixres:
            #assert resnet18
            model.eval()
            if self.arch == 'resnet18':
                model.module.layer4[1].bn2.train()
            elif self.arch == 'wideres':
                model.module.bn1.train()
            else:
                raise NotImplementedError
        else:
            model.train()
        steps_per_epoch = len(self.train_loader)
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm)
        for i, (input, target, _) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True)

            smoothed_targets = self.smooth_one_hot(target)
            assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            r = np.random.rand(1)
            if alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            elif beta > 0 and r < cutmix_prob: # Cutmix augmentation
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                output = model(input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = self.cross_entropy(output, smoothed_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target).float().mean()
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                if callback is not None:
                    callback.scalar('train_loss', i / steps_per_epoch + epoch, losses.avg, title='Train loss')
                    callback.scalar('@1', i / steps_per_epoch + epoch, top1.avg, title='Train Accuracy')

        # Track learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        if callback is not None:
            callback.scalar('lr', epoch, current_lr, title='Learning rate')

    @trainer_ingredient.capture
    def smooth_one_hot(self, targets, label_smoothing):
        assert 0 <= label_smoothing < 1
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=self.device)
            new_targets.fill_(label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - label_smoothing)
        return new_targets

    @trainer_ingredient.capture
    def meta_val(self, model, meta_val_way, meta_val_shot,
                 disable_tqdm, callback, epoch, train_mean=None):
        top1 = AverageMeter()
        model.eval()

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target, _) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                output = model(inputs, feature=True)[0].cuda(0)
                if train_mean is not None:
                    output = output - train_mean
                train_out = output[:meta_val_way * meta_val_shot]
                train_label = target[:meta_val_way * meta_val_shot]
                test_out = output[meta_val_way * meta_val_shot:]
                test_label = target[meta_val_way * meta_val_shot:]
                train_out = train_out.reshape(meta_val_way, meta_val_shot, -1).mean(1)
                train_label = train_label[::meta_val_shot]
                prediction = self.metric_prediction(train_out, test_out, train_label)
                acc = (prediction == test_label).float().mean()
                top1.update(acc.item())
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))

        if callback is not None:
            callback.scalar('val_acc', epoch + 1, top1.avg, title='Val acc')
        return top1.avg

    @trainer_ingredient.capture
    def metric_prediction(self, gallery, query, train_label, meta_val_metric):
        gallery = gallery.view(gallery.shape[0], -1)
        query = query.view(query.shape[0], -1)
        distance = get_metric(meta_val_metric)(gallery, query)
        predict = torch.argmin(distance, dim=1)
        predict = torch.take(train_label, predict)
        return predict


class Pretrainer:
    @trainer_ingredient.capture
    def __init__(self, device, ex, temp):
        self.train_loader = get_dataloader(split='train', shuffle=True, pretrain=True).get()
        self.device = device
        self.num_classes = ex.current_run.config['model']['num_classes']
        self.arch = ex.current_run.config['model']['arch']

        self.criterion = SupConLoss(temperature=temp).to(self.device)

    @trainer_ingredient.capture
    def train(self, epoch, scheduler, print_freq, disable_tqdm, callback, model, warmup_to, warm_epochs,optimizer):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader)
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm)
        for i, (input, target, _) in enumerate(tqdm_train_loader):
            input = torch.cat([input[0], input[1]], dim=0)
            input, target = input.to(self.device), target.to(self.device, non_blocking=True)
            bsz = target.shape[0]

            warmup_learning_rate(warm_epochs, 0.01, warmup_to, epoch+1, i, len(self.train_loader), optimizer)

            # skip warmp-up learning rate

            # Forward pass
            output = model(input)
            f1, f2 = torch.split(output, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            if not disable_tqdm:
                tqdm_train_loader.set_description('LR {:.2f}'.format(current_lr))

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' .format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses))

        # Track learning rate
        #for param_group in optimizer.param_groups:
        #    current_lr = param_group['lr']
        #if callback is not None:
        #    callback.scalar('lr', epoch, current_lr, title='Learning rate')

