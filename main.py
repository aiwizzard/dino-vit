import argparse
import yaml
import torch
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    ImageAugmentation,
    ProjectionHead,
    MultiCropWrapper,
    DINOLoss,
    clip_gradients,
)
from model import VisionTransformer


def main():
    parser = argparse.ArgumentParser("DINO_VIT")
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        type=str,
        help="Please specify the path to the config file",
    )
    args = parser.parse_args()
    with open(args.config_path) as file:
        config = yaml.load(file, yaml.FullLoader)

    device = torch.device(config['device'])
    transform = ImageAugmentation(
        config['global_crop_scale'], config['local_crop_scale'], config['n_local_crops']
    )
    dataset = ImageFolder(config['data_path'], transform=transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    # Specify the models
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
    head = ProjectionHead(config['embed_size'], config['out_dim'])
    teacher = MultiCropWrapper(backbone, head).to(device)
    student = MultiCropWrapper(backbone, head).to(device)

    teacher.load_state_dict(student.state_dict())

    for param in teacher.parameters():
        param.require_grads = False

    dino_loss = DINOLoss(
        config['out_dim'], config['teacher_temp'], config['student_temp'], config['center_momentum']
    ).to(device)
    lr = 0.0005 * config['batch_size'] / 256
    optimizer = optim.AdamW(
        student.parameters(), lr=lr, weight_decay=config['weight_decay']
    )

    for epoch in range(config['n_epochs']):
        train(
            epoch, data_loader, student, teacher, dino_loss, optimizer, device, config
        )


def train(epoch, data_loader, student, teacher, dino_loss, optimizer, device, config):
    with tqdm(total=len(data_loader), desc=f"Epoch: {epoch + 1}") as pbar:
        for images, _ in data_loader:
            # Train
            images = [img.to(device) for img in images]
            teacher_output = teacher(images[:2])
            student_ouptut = student(images)
            loss = dino_loss(teacher_output, student_ouptut).to(device)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, clip=config['clip_grad'])
            optimizer.step()

            with torch.no_grad():
                # Note: Cosine Scheduler is ignored for now, might add later
                for t_params, s_params in zip(teacher.parameters(), student.parameters()):
                    t_params.data.mul_(config['momentum_teacher'])
                    t_params.data.add_((1 - config['momentum_teacher']) * s_params.detach().data)
            pbar.update(1)
            pbar.set_postfix_str(f"loss: {loss:.5f}")

    torch.save(
        {
            "epoch": epoch + 1,
            "student": student.parameters(),
            "teacher": teacher.parameters(),
            "optimizer": optimizer.state_dict(),
            "loss": dino_loss.state_dict(),
            "config": config,
        },
        f"{config['model_path']}/{config['model_name']}.pth",
    )

if __name__ == '__main__':
    main()