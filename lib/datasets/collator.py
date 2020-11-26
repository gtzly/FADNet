# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from lib.utils.image_list import to_image_list


class Collator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0])
        meta_data = transposed_batch[1]
        img_ids = transposed_batch[2]
        return dict(images=images,
                    meta_data=meta_data,
                    img_ids=img_ids)
