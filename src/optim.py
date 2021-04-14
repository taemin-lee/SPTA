import math
import torch.optim
from sacred import Ingredient
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

optim_ingredient = Ingredient('optim')


@optim_ingredient.config
def config():
    gamma = 0.1
    lr = 0.1
    lr_stepsize = 30
    nesterov = False
    weight_decay = 1e-4
    optimizer_name = 'SGD'
    scheduler = None
    pretraining_lr = 0.5

class get_scheduler:
    @optim_ingredient.capture
    def __init__(self, epochs, batches, optimizer, gamma, lr_stepsize, scheduler):
        SCHEDULER = {'step': StepLR(optimizer, lr_stepsize, gamma),
                     'multi_step': MultiStepLR(optimizer, milestones=[int(.5 * epochs), int(.75 * epochs)],
                                               gamma=gamma),
                     'cosine': CosineAnnealingLR(optimizer, batches * epochs, eta_min=1e-9),
                     None: None}
        self.scheduler = SCHEDULER[scheduler]

    def get(self):
        return self.scheduler

class get_optimizer:
    @optim_ingredient.capture
    def __init__(self, module, optimizer_name, nesterov, lr, weight_decay, pretraining_lr, pretrain=False):
        OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=pretraining_lr if pretrain else lr, momentum=0.9, weight_decay=weight_decay,
                                            nesterov=nesterov),
                     'Adam': torch.optim.Adam(module.parameters(), lr=pretraining_lr if pretrain else lr)}
        self.optimizer = OPTIMIZER[optimizer_name]

    def get(self):
        return self.optimizer


def adjust_learning_rate(learning_rate, lr_decay_rate, epochs, optimizer, epoch):
    lr = learning_rate
    eta_min = lr * (lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(warm_epochs, warmup_from, warmup_to, epoch, batch_id, total_batches, optimizer):
    if epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


