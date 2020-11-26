from . import transforms as T


def build_transforms(cfg):
    normalize_transform = T.Normalize(
        mean=cfg.input_mean, std=cfg.input_std, to_bgr=True
    )

    transform = T.Compose(
        [
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
