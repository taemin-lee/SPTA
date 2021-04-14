from src.main import Mainer
import torch.backends.cudnn as cudnn
import tempfile
import argparse

cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--way', type=int, default=10, choices=[10, 20])
args = parser.parse_args()

global_config = {}
global_config["eval.meta_val_way"] = args.way

global_config["dataset.path"] = "/SSD/tim/data/mini_imagenet"
global_config["dataset.split_dir"] = "./split/mini"
global_config["model.num_classes"] = 64

global_config["model.arch"] = "resnet18"
global_config["ckpt_path"] = "./checkpoints/mini/softmax/resnet18"

global_config["eval.init_mode"] = "proto_rect"
global_config["tim.iter"] = 150
global_config["evaluate"] = True
global_config["dataset.imsize"] = 84
global_config["dataset.num_workers"]=20
global_config["eval.overwrite"] = True
global_config["eval.model_tag"] = 'best'
global_config["eval.method"] = "tim_adm"

global_config["eval.meta_test_iter"] = 100000 // global_config["eval.meta_val_way"]
split = global_config["eval.meta_val_way"] // 10
results = []

if global_config["eval.method"] == "tim_gd":
    global_config["tim.iter"] = 1000

def trainable(config):
    full_config = {**global_config, **config}
    r = Mainer.ex.run(config_updates=full_config)
    print("result: {} (type: {})".format(r.result, type(r.result)))
    results.append(r.result)

if __name__ == '__main__':
    config = {}
    if split > 1:
        for i in range(split):
            config['seed'] = i * 1010
            trainable(config)

        print(results)

        shot1 = [r[0] for r in results]
        shot5 = [r[1] for r in results]

        def get_mean(x):
            return sum(x) / len(x)

        print('result avg:', get_mean(shot1), get_mean(shot5))
    else:
        trainable(config)
