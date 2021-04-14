import os
import random
import math
import sacred
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from visdom_logger import VisdomLogger
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from src.utils import warp_tqdm, save_checkpoint
from src.trainer import Trainer, trainer_ingredient, Pretrainer
from src.eval import Evaluator
from src.eval import eval_ingredient
from src.tim import tim_ingredient
from src.optim import optim_ingredient, get_optimizer, get_scheduler, adjust_learning_rate
from src.datasets.ingredient import dataset_ingredient
from src.models.ingredient import get_model, model_ingredient
from src.models.SupCon import SupConNet

class Mainer:
    ex = sacred.Experiment('FSL training',
                           ingredients=[trainer_ingredient, eval_ingredient,
                                        optim_ingredient, dataset_ingredient,
                                        model_ingredient, tim_ingredient])
    # Filter backspaces and linefeeds
    SETTINGS.CAPTURE_MODE = 'sys'
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    
    
    @ex.config
    def config():
        ckpt_path = os.path.join('checkpoints')
        seed = 2020
        pretrain = False
        resume = False
        evaluate = False
        make_plot = False
        epochs = 90
        disable_tqdm = False
        visdom_port = None
        print_runtime = False
        cuda = True
        ray = False
        fixres = False
        fixres_path = os.path.join('checkpoints')
        pretraining = False
        pretraining_epochs = 350
        warm_epochs = 10

    #def main(seed):
    #    return [seed, seed]
    @ex.automain
    def main(seed, pretrain, resume, evaluate, print_runtime,
             epochs, disable_tqdm, visdom_port, ckpt_path,
             make_plot, cuda, ray, fixres, fixres_path, pretraining, pretraining_epochs, warm_epochs):
        device = torch.device("cuda" if cuda else "cpu")
        callback = None if visdom_port is None else VisdomLogger(port=visdom_port)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
        torch.cuda.set_device(0)
        # create model
        print("=> Creating model '{}'".format(Mainer.ex.current_run.config['model']['arch']))
        model = get_model().get()
    
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if pretraining:
            pretraining_model = SupConNet(model, name=Mainer.ex.current_run.config['model']['arch'])
            pretraining_model = torch.nn.DataParallel(pretraining_model).cuda()
            pretrainer = Pretrainer(device=device, ex=Mainer.ex)
            pretrainer_optimizer = get_optimizer(pretraining_model, pretrain=True).get()
            pretrainer_scheduler = get_scheduler(optimizer=pretrainer_optimizer,
                                      batches=len(pretrainer.train_loader),
                                      epochs=pretraining_epochs).get()
            tqdm_loop = warp_tqdm(list(range(0, pretraining_epochs)),
                                  disable_tqdm=disable_tqdm)

            eta_min = Mainer.ex.current_run.config['optim']['pretraining_lr'] * (0.1 ** 3)
            warmup_to = eta_min + (Mainer.ex.current_run.config['optim']['pretraining_lr'] - eta_min) * (
                    1 + math.cos(math.pi * warm_epochs / pretraining_epochs)) / 2
            for epoch in tqdm_loop:
                # Do one epoch
                adjust_learning_rate(Mainer.ex.current_run.config['optim']['pretraining_lr'], 0.1, pretraining_epochs, pretrainer_optimizer, epoch)
                pretrainer.train(model=pretraining_model, optimizer=pretrainer_optimizer, epoch=epoch,
                              scheduler=pretrainer_scheduler, disable_tqdm=disable_tqdm,
                              callback=callback, warmup_to=warmup_to, warm_epochs=warm_epochs)
                save_checkpoint(state={'epoch': epoch + 1,
                                       'arch': Mainer.ex.current_run.config['model']['arch'],
                                       'state_dict': pretraining_model.state_dict(),
                                       #'optimizer': optimizer.state_dict()
                                       },
                                is_best=False,
                                folder=pretrain)
                #if pretrainer_scheduler is not None:
                #    pretrainer_scheduler.step()

        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(device)
        optimizer = get_optimizer(model).get()

        if pretrain:
            pretrain = os.path.join(pretrain, 'checkpoint.pth.tar')
            if os.path.isfile(pretrain):
                print("=> loading pretrained weight '{}'".format(pretrain))
                checkpoint = torch.load(pretrain)
                model_dict = model.state_dict()
                params = checkpoint['state_dict']
                params = {k.replace('.encoder', ''): v for k, v in params.items() if '.encoder' in k}
                params = {k: v for k, v in params.items() if k in model_dict}
                model_dict.update(params)
                model.load_state_dict(model_dict)
            else:
                print('[Warning]: Did not find pretrained model {}'.format(pretrain))
    
        if resume:
            resume_path = ckpt_path + '/checkpoint.pth.tar' 
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                # scheduler.load_state_dict(checkpoint['scheduler'])
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print('[Warning]: Did not find checkpoint {}'.format(resume_path))
        elif fixres:
            resume_path = fixres_path + '/model_best.pth.tar' 
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                #start_epoch = checkpoint['epoch']
                #best_prec1 = checkpoint['best_prec1']
                start_epoch = 0
                best_prec1 = -1
                # scheduler.load_state_dict(checkpoint['scheduler'])
                model.load_state_dict(checkpoint['state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print('[Warning]: Did not find checkpoint {}'.format(resume_path))
        else:
            start_epoch = 0
            best_prec1 = -1
    
        cudnn.benchmark = True
    
        # Data loading code
        evaluator = Evaluator(device=device, ex=Mainer.ex, disable_tqdm=disable_tqdm)
        if evaluate:
            results = evaluator.run_full_evaluation(model=model,
                                                    model_path=ckpt_path,
                                                    callback=callback)
            return results
    
        # If this line is reached, then training the model
        trainer = Trainer(device=device, ex=Mainer.ex)
        scheduler = get_scheduler(optimizer=optimizer,
                                  batches=len(trainer.train_loader),
                                  epochs=epochs).get()
        tqdm_loop = warp_tqdm(list(range(start_epoch, epochs)),
                              disable_tqdm=disable_tqdm)
        for epoch in tqdm_loop:
            # Do one epoch
            trainer.train(model=model, optimizer=optimizer, epoch=epoch,
                          scheduler=scheduler, disable_tqdm=disable_tqdm,
                          callback=callback, fixres=fixres)
    
            # Evaluation on validation set
            if (epoch) % trainer.meta_val_interval == 0:
                prec1 = trainer.meta_val(model=model, disable_tqdm=disable_tqdm,
                                         epoch=epoch, callback=callback)
                if ray:
                    from ray import tune
                    tune.report(mean_accuracy=prec1)
                print('Meta Val {}: {}'.format(epoch, prec1))
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if not disable_tqdm:
                    tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))
    
            # Save checkpoint
            save_checkpoint(state={'epoch': epoch + 1,
                                   'arch': Mainer.ex.current_run.config['model']['arch'],
                                   'state_dict': model.state_dict(),
                                   'best_prec1': best_prec1,
                                   'optimizer': optimizer.state_dict()},
                            is_best=is_best,
                            folder=ckpt_path)
            if scheduler is not None:
                scheduler.step()
    
        if epochs == 0:
            save_checkpoint(state={'epoch': 0,
                                   'arch': Mainer.ex.current_run.config['model']['arch'],
                                   'state_dict': model.state_dict(),
                                   },
                                   #'best_prec1': best_prec1,
                                   #'optimizer': optimizer.state_dict()},
                            is_best=True,
                            folder=ckpt_path)
        # Final evaluation on test set
        results = evaluator.run_full_evaluation(model=model, model_path=ckpt_path, callback=callback)
        results.insert(0, best_prec1)
        return results
