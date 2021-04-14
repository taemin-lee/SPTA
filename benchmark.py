from src.main import Mainer
import torch.backends.cudnn as cudnn
import tempfile
import argparse

cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=str, default='baseline', choices=[
    'baseline', 'reduced'])
args = parser.parse_args()

global_config = {}
global_config["dataset.path"] = "/SSD/tim/data/mini_imagenet"
global_config["dataset.split_dir"] = "./split/mini"
global_config["evaluate"] = True
global_config["dataset.imsize"] = 84
global_config["dataset.num_workers"]=20

global_config["eval.overwrite"] = False
global_config["eval.model_tag"] = 'best'
global_config["eval.method"] = "tim_adm"

test_baseline=args.setting
if test_baseline == 'baseline':
    global_config["eval.init_mode"] = None
    global_config["model.arch"] = "resnet18"
    global_config["tim.iter"] = 150
else:
    global_config["eval.init_mode"] = "proto_rect"
    global_config["model.arch"] = "mobilenet"
    global_config["tim.iter"] = 10

global_config["ckpt_path"] = "./checkpoints/mini/softmax/" + global_config["model.arch"]
global_config["eval.overwrite"] = True
global_config["eval.meta_test_iter"] = 1
global_config["eval.used_set"] = 'bench'
global_config["dataset.batch_size"] = 100
global_config["cuda"] = True
global_config["disable_tqdm"] = True
global_config["eval.benchmark"] = True

def trainable(config):
    full_config = {**global_config, **config}
    r = Mainer.ex.run(config_updates=full_config)
    print("result: {} (type: {})".format(r.result, type(r.result)))

if __name__ == '__main__':
    config = {}
    trainable(config)

