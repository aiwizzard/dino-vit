import argparse
import yaml
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# to fix the issue AttributeError: module 'tensorflow._api.v2.io.gfile' 
# has no attribute 'get_filesystem'
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from tqdm import tqdm
from utils import (
    ImageAugmentation,
    ProjectionHead,
    MultiCropWrapper,
    DINOLoss,
    clip_gradients,
)
from model import VisionTransformer
from evaluate import compute_embedding, compute_knn


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

    labels_path = config["labels_path"]
    logging_path = config["logging_path"]

    with open(labels_path, "r") as file:
        label_mapping = json.load(file)

    device = torch.device(config["device"])
    transform = ImageAugmentation(
        config["global_crop_scale"], config["local_crop_scale"], config["n_local_crops"]
    )

    transform_wna = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset = ImageFolder(config["data_path_train"], transform=transform)
    dataset_wna = ImageFolder(config["data_path_train"], transform=transform_wna)
    dataset_val_wna = ImageFolder(config["data_path_val"], transform=transform_wna)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    data_loader_train_wna = DataLoader(
        dataset_wna,
        batch_size=config["batch_size_eval"],
        num_workers=config["num_workers"],
    )

    data_loader_val_wna = DataLoader(
        dataset_val_wna,
        batch_size=config["batch_size_eval"],
        num_workers=config["num_workers"],
    )

    data_loader_val_wna_subset = DataLoader(
        dataset_val_wna,
        batch_size=config["batch_size_eval"],
        sampler=SubsetRandomSampler(list(range(0, len(dataset_val_wna), 50))),
        num_workers=config["num_workers"],
    )

    # Tensorboard Summarywriter for logging
    writer = SummaryWriter(logging_path)
    writer.add_text("Configuration", json.dumps(config))

    # Specify the models
    backbone = VisionTransformer(
        config["n_classes"],
        config["depth"],
        config["image_size"],
        config["in_channels"],
        config["embed_size"],
        config["patch_size"],
        config["head"],
        config["hidden_size"],
        config["dropout_rate"],
    )
    head = ProjectionHead(config["embed_size"], config["out_dim"])
    teacher = MultiCropWrapper(backbone, head).to(device)
    student = MultiCropWrapper(backbone, head).to(device)

    teacher.load_state_dict(student.state_dict())

    for param in teacher.parameters():
        param.require_grads = False

    dino_loss = DINOLoss(
        config["out_dim"],
        config["teacher_temp"],
        config["student_temp"],
        config["center_momentum"],
    ).to(device)
    lr = 0.0005 * config["batch_size"] / 256
    optimizer = optim.AdamW(
        student.parameters(), lr=lr, weight_decay=config["weight_decay"]
    )

    dataset_len = len(dataset)

    # to save the configs with torch.save()
    config_list = []
<<<<<<< HEAD
    for item in config.items():
=======
    for item in config.item():
>>>>>>> 757d86e1d8cc3b8f30888d6c2a37fa39afc03c45
        config_list.append(item)

    for epoch in range(config["n_epochs"]):
        train(
            epoch,
            data_loader,
            student,
            teacher,
            dino_loss,
            optimizer,
            device,
            config,
            config_list,
            writer,
            dataset_len,
            label_mapping,
            data_loader_train_wna,
            data_loader_val_wna,
            data_loader_val_wna_subset,
        )


def train(
    epoch,
    data_loader,
    student,
    teacher,
    dino_loss,
    optimizer,
    device,
    config,
    config_list,
    writer,
    dataset_len,
    label_mapping,
    data_loader_train_wna,
    data_loader_val_wna,
    data_loader_val_wna_subset,
):
    with tqdm(total=len(data_loader), desc=f"Epoch: {epoch + 1}") as pbar:
        n_batches = dataset_len // config["batch_size"]
        n_steps = 0
        best_acc = 0
        for images, _ in data_loader:
            student.train()
            # Train
            images = [img.to(device) for img in images]
            teacher_output = teacher(images[:2])
            student_ouptut = student(images)
            loss = dino_loss(teacher_output, student_ouptut).to(device)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, clip=config["clip_grad"])
            optimizer.step()

            with torch.no_grad():
                # Note: Cosine Scheduler is ignored for now, might add later
                for t_params, s_params in zip(
                    teacher.parameters(), student.parameters()
                ):
                    t_params.data.mul_(config["momentum_teacher"])
                    t_params.data.add_(
                        (1 - config["momentum_teacher"]) * s_params.detach().data
                    )

            if n_steps % config["logging_freq"] == 0:
                student.eval()

                embs, imgs, labels_ = compute_embedding(
                    student.backbone, data_loader_val_wna_subset
                )
                writer.add_embedding(
                    embs,
                    metadata=[label_mapping[l] for l in labels_],
                    label_img=imgs,
                    global_step=n_steps,
                    tag="embeddings",
                )

                # KNN
                current_acc = compute_knn(
                    student.backbone, data_loader_train_wna, data_loader_val_wna
                )
                writer.add_scalar("knn-accuracy", current_acc, n_steps)
                if current_acc > best_acc:
                    best_acc = current_acc

            pbar.update(1)
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            
            writer.add_scalar("train_loss", loss, n_steps)

            n_steps += 1

    torch.save(
        {
            "epoch": epoch + 1,
            "student": student.parameters(),
            "teacher": teacher.parameters(),
            "optimizer": optimizer.state_dict(),
            "loss": dino_loss.state_dict(),
            "config": config_list,
        },
        f"{config['model_path']}/{config['model_name']}.pth",
    )


if __name__ == "__main__":
    main()
