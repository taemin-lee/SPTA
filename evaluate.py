from src.main import Mainer
import torch.backends.cudnn as cudnn
import tempfile
import argparse

cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mini', choices=[
    'mini', 'tiered', 'cub'])
parser.add_argument('--network', type=str, default='resnet18', choices=[
    'mobilenet', 'resnet18', 'wideres'])
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

global_config["model.arch"] = args.network
global_config["ckpt_path"] = "./checkpoints/" + args.dataset + "/softmax/" + args.network

global_config["eval.init_mode"] = "proto_rect"
#global_config["eval.init_mode"] = None
global_config["tim.iter"] = 150
global_config["evaluate"] = True
global_config["dataset.imsize"] = 84
global_config["dataset.num_workers"]=20
global_config["eval.overwrite"] = True
global_config["eval.model_tag"] = 'best'
global_config["eval.method"] = "tim_adm"

def trainable(config):
    full_config = {**global_config, **config}
    r = Mainer.ex.run(config_updates=full_config)
    print("result: {} (type: {})".format(r.result, type(r.result)))

if __name__ == '__main__':
    if args.dataset != 'mini' and args.network == 'wideres':
        raise NotImplementedError
    config = {}
    trainable(config)
