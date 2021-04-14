from sacred import Ingredient
from src.datasets.loader import DatasetFolder
from src.datasets.sampler import CategoriesSampler
from src.datasets.transform import with_augment, without_augment
from src.datasets.transform_fixres import get_transforms
from src.datasets.transform_supcon import set_transform
from torch.utils.data import DataLoader


dataset_ingredient = Ingredient('dataset')
@dataset_ingredient.config
def config():
    batch_size = 256
    pretraining_batch_size = 512
    enlarge = True
    num_workers = 4
    disable_random_resize = False
    jitter = False
    path = 'data'
    split_dir = None
    imsize = 84
    rand_augment = False
    ra_n = 1
    ra_m = 5

class get_dataloader:
    @dataset_ingredient.capture
    def __init__(self, split, enlarge, num_workers, batch_size, disable_random_resize,
                       path, split_dir, jitter, aug=False, shuffle=True, out_name=False,
                       sample=None, imsize=84, rand_augment=False, ra_n=1, ra_m=5, fixres=False, 
                       pretrain=False, pretraining_batch_size=512):
        # sample: iter, way, shot, query
        if pretrain:
            transform = set_transform(size=imsize)
        elif fixres:
            transform = get_transforms(input_size=imsize, test_size=imsize, 
                    kind='full', crop=True, need=('val'), backbone=None, 
                    rand_augment=rand_augment, ra_n=ra_n, ra_m=ra_m, jitter=jitter)['val']
        elif aug:
            transform = with_augment(imsize, disable_random_resize=disable_random_resize,
                                     jitter=jitter, rand_augment=rand_augment, ra_n=ra_n, ra_m=ra_m)
        else:
            transform = without_augment(imsize, enlarge=enlarge)
        sets = DatasetFolder(path, split_dir, split, transform, out_name=out_name)
        if sample is not None:
            sampler = CategoriesSampler(sets.labels, *sample)
            loader = DataLoader(sets, batch_sampler=sampler,
                                num_workers=num_workers, pin_memory=False)
        else:
            loader = DataLoader(sets, batch_size=pretraining_batch_size if pretrain else batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=False)
        self.loader = loader

    def get(self):
        return self.loader

