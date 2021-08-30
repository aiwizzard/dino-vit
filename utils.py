import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image


class RandomGaussianBlur:
    r"""RandomGaussianBlur class

    Apply Gaussian Blur to the PIL image
    """

    def __init__(self, prob=0.5, min_radius=0.1, max_radius=2.0) -> None:
        self.prob = prob
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, img) -> Image:
        if torch.rand(1) < self.prob:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.min_radius, self.max_radius)
                )
            )
        return img


class Solarization:
    r"""Solarization Class

    Apply solarization to the PIL image.
    Solarization is the effect of tone reversal.
    """

    def __init__(self, prob) -> None:
        self.prob = prob

    def __call__(self, img: Image) -> Image:
        if torch.rand(1) < self.prob:
            return ImageOps.solarize(img)
        return img


class ImageAugmentation:
    r"""ImageAugmentation Class

    Apply various transformations to the PIL Image
    """

    def __init__(self, global_crop_scales, local_crop_scales, n_local_crops) -> None:
        self.n_local_crops = n_local_crops

        flip_and_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.8, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ]
                ),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=global_crop_scales,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(prob=1.0),  # not random
                normalize,
            ]
        )

        self.global_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=global_crop_scales,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96,
                    scale=local_crop_scales,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(prob=0.5),
                normalize,
            ]
        )

    def __call__(self, img):
        crops = []
        crops.append(self.global_transform1(img))
        crops.append(self.global_transform2(img))
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(img))
        return crops


class ProjectionHead(nn.Module):
    r"""ProjectionHead Class

    This consist of 3 layer multi perceptron followed by l2 weight
    normalization and a weight normalized fully connected layer.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head) -> None:
        super(MultiCropWrapper, self).__init__()
        backbone.head = nn.Identity()  # disable the original head
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # get the counts of the same resolution images
        _, counts = torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]), return_counts=True
        )
        n_crops = len(x)
        idx_crops = torch.cumsum(counts, 0)
        start_idx = 0
        output = torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output = torch.cat((output, _out))
            start_idx = end_idx
        out = self.head(output)
        return out.chunk(n_crops)


class DINOLoss(nn.Module):
    r"""DINOLoss Class

    Compute the loss.
    """

    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum) -> None:
        super(DINOLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = [
            F.log_softmax(s / self.student_temp, dim=-1) for s in student_output
        ]
        teacher_out = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_output
        ]

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t in enumerate(teacher_out):
            for s_idx, s in enumerate(student_out):
                # Skip for the same image
                if t_idx == s_idx:
                    continue
                loss = torch.sum(-t * s, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        r"""Update center for teacher output

        Exponential moving average as described in the paper
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )


def clip_gradients(model, clip):
    r"""Clip gradients.

    This function will normalize the gradients.
    Thus avoid gradient Explotion
    """
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                param.grad.data.mul_(clip_coef)
