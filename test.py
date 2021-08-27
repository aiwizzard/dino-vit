# This is only for testing if forward pass work

import yaml
import torch
from PIL import Image
from utils import ImageAugmentation, ProjectionHead, MultiCropWrapper
from model import VisionTransformer


def main():

    with open("config.yaml") as file:
        config = yaml.load(file, yaml.FullLoader)
    img_path = "path to image"
    img = Image.open(img_path).convert("RGB")

    transform = ImageAugmentation(
        config["global_crop_scale"], config["local_crop_scale"], config["n_local_crops"]
    )

    img_list = transform(img)
    img_list = [img.unsqueeze(0).to(torch.float32) for img in img_list]

    backbone = VisionTransformer(
        config['n_classes'],
        config['depth'],
        config['image_size'],
        config['in_channels'],
        config['embed_size'],
        config['patch_size'],
        config['head'],
        config['hidden_size'],
        config['dropout_rate'],
    )

    model = MultiCropWrapper(backbone, ProjectionHead(config['embed_size'], config['out_dim']))
    out = model(img_list)

    print("Forward pass Successful")

if __name__ == '__main__':
    main()

