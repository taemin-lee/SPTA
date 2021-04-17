from src.main import Mainer
import torch.backends.cudnn as cudnn
import tempfile
import argparse

cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mini', choices=[
    'mini', 'tiered', 'cub'])
parser.add_argument('--network', type=str, default='resnet18', choices=[
    'mobilenet', 'resnet10', 'resnet12', 'resnet18', 'wideres'])
parser.add_argument('--pretraining_batch_size', type=int, default=1024)
args = parser.parse_args()

dataset=args.dataset
global_config = {}
if dataset =='mini':
    global_config["dataset.path"] = "/SSD/tim/data/mini_imagenet"
    global_config["dataset.split_dir"] = "./split/mini"
    global_config["model.num_classes"] = 64
elif dataset == 'tiered':
    global_config["dataset.path"] = "/SSD/tim/data/tiered_imagenet/data"
    global_config["dataset.split_dir"] = "./split/tiered"
    global_config["model.num_classes"] = 351
elif dataset == 'cub':
    global_config["dataset.path"] = "/SSD/tim/data/cub/CUB_200_2011/images"
    global_config["dataset.split_dir"] = "./split/cub"
    global_config["model.num_classes"] = 100

global_config["visdom_port"] = None
global_config["dataset.jitter"] = True
global_config["optim.scheduler"] = "multi_step"
global_config["trainer.label_smoothing"]=0.1
global_config["dataset.num_workers"]=40
global_config["eval.method"] = "tim_adm"

global_config["ckpt_path"] = "./checkpoints_resnet18_resume/cub/softmax"
global_config["pretrain"] = "./checkpoints_resnet18/cub/softmax"
global_config["model.arch"] = args.network
global_config["dataset.batch_size"] = 256
global_config["dataset.pretraining_batch_size"] = args.pretraining_batch_size
global_config["pretraining"] = True

global_config["epochs"] = 5
global_config["pretraining_epochs"] = 1000
global_config["optim.lr"] = 0.05
global_config["optim.pretraining_lr"] = 0.5
global_config["eval.init_mode"] = "proto_rect"

if global_config["model.arch"] == "wideres":
    global_config["dataset.batch_size"] = 64
    global_config["dataset.pretraining_batch_size"] = 128


def trainable(config):
    full_config = {**global_config, **config}
    r = Mainer.ex.run(config_updates=full_config)
    print("result: {} (type: {})".format(r.result, type(r.result)))

if __name__ == '__main__':
    config = {}
    trainable(config)
