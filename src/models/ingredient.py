from sacred import Ingredient
from src.models import *

model_ingredient = Ingredient('model')
@model_ingredient.config
def config():
    arch = 'resnet18'
    num_classes = 64

class get_model:
    @model_ingredient.capture
    def __init__(self, arch, num_classes):
        assert arch == 'mobilenet' or \
                arch == 'resnet10' or \
                arch == 'resnet18' or \
                arch == 'wideres'
        self.arch = arch
        self.num_classes = num_classes

        if arch == 'mobilenet':
            self.model = mobilenet(num_classes=self.num_classes)
        elif arch == 'resnet10':
            self.model = resnet10(num_classes=self.num_classes)
        elif arch == 'resnet18':
            self.model = resnet18(num_classes=self.num_classes)
        elif arch == 'wideres':
            self.model = wideres(num_classes=self.num_classes)

    def get(self):
        return self.model

