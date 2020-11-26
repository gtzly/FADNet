import os
import csv
import torch
import time
import sys
sys.path.append(os.getcwd())
from collections import OrderedDict
from config.KITTI3D import get_configs
from tqdm import tqdm
from lib.networks.fadnet import FADNet
from lib.datasets.kitti import KITTI
from lib.transforms import build_transforms
from lib.datasets.collator import Collator

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

def main():
    cfg = get_configs()

    model = FADNet(cfg)
    device = torch.device("cuda")
    model.to(device)

    state_dict=torch.load(cfg.pretrained_model,map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]]=v
    model.load_state_dict(new_state_dict)

    transforms = build_transforms(cfg)
    data_loader_val = torch.utils.data.DataLoader(
            KITTI(cfg,transforms=transforms),
            num_workers=1,
            collate_fn=Collator(),
        )

    model.eval()
    results_dict = {}
    total_time=0
    for i, batch in enumerate(tqdm(data_loader_val)):
        images, meta_data, image_ids = batch["images"], batch["meta_data"], batch["img_ids"]
        images = images.to(torch.device("cuda"))
        with torch.no_grad():
            start_time=time.time()
            output = model(images, meta_data)
            total_time+=(time.time()-start_time)
            output = output.to(torch.device("cpu"))
        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    print("Avg Running Speed: {} FPS.".format((i+1)/total_time))

    output_folder = os.path.join(cfg.output_dir, "test")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cls_id_dict = {0: 'Car',1: 'Cyclist',2: 'Pedestrian'}
    for image_id, prediction in results_dict.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(output_folder, predict_txt)
        with open(predict_txt, 'w', newline='') as f:
            w = csv.writer(f, delimiter=' ', lineterminator='\n')
            if len(prediction) == 0:
                w.writerow([])
            else:
                for p in prediction:
                    p = p.numpy()
                    p = p.round(4)
                    type = cls_id_dict[int(p[0])]
                    row = [type, 0, 0] + p[1:].tolist()
                    w.writerow(row)

        check_last_line_break(predict_txt)

if __name__ == '__main__':
    main()
