""" PyTorch DPN hubconf.py

## Users can get this published model by calling:
hub_model = hub.load(
    'rwightman/pytorch-dpn-pretrained:master', # repo_owner/repo_name:branch
    'dpn92', # entrypoint
    pretrained=True) # kwargs for callable
"""
dependencies = ['torch', 'math', 'torchvision', 'scipy']

from dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from sotabench.image_classification import imagenet

import torchvision.transforms as transforms
import PIL


def benchmark():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    imagenet.benchmark(
        model=dpn68(pretrained=True),
        paper_model_name='DPN-68',
        paper_arxiv_id='1707.01629',
        paper_pwc_id='dual-path-networks',
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1
    )
