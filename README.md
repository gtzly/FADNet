# FADNet
Implementation for Monocular 3D Object Detection with Sequential Feature Association and Depth Hint Augmentation.
For now, we only release the testing code



# Environment
- NVIDIA RTX 2080Ti
- Ubuntu=16.04, cuda=10.0
- python=3.6
- torch==1.6.0, torchvision==0.7.0
- tqdm, easydict, scikit-image

# Preparation
- Download [KITTI3D dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the images and labels as follows:  
<pre>
--kitti  
    --training  
        --calib  
        --image_2   
        --label_2  
    --testing  
        --calib   
        --image_2  
</pre>

- Clone the repository.
`$ git clone https://github.com/gtzly/FADNet`

- Establish a soft link to KITTI3D dataset.  
`$ cd FADNet`  
`$ ln -s path_to_kitti ./data`

- Install DCNv2.  
`$ cd lib/layers/DCNv2`  
`$ bash ./make.sh`  

- Download the pretrained [model](https://drive.google.com/file/d/1xaqoG8WgJS5VC-5HsCN0jVPH4LpHlVhH/view?usp=sharing) and put it under a folder named as 'checkpoint'.  
`$ cd ../../..`
`$ mkdir checkpoint`
`$ mv final.pth checkpoint`

# Getting Started  
- Running FADNet on KITTI3D test set with the pretrained model.
`$ python scripts/test.py`

- Visualize the detection results.  
`$ python scripts/visualize.py`

# Evaluation
- The performance on KITTI3D test set:  

    | Benchmark | Easy | Moderate | Hard |
    | :--------: | :----: | :----: | :----: |
    | Car (Detection) |96.15 % |	90.49 % | 80.71 % |
    | Car (Orientation) | 95.89 % |	89.84 % |	79.98 % |
    | Car (3D Detection) | 16.37 % |	9.92 % |	8.05 % |
    | Car (Bird's Eye View) | 23.00 % |	14.22 % |	12.56 % |
*Note: The above statistics are accessible on the [KIITI official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Readers can submit the test results obtained by the provided pretrained model to the KITTI3D benchmark to verify the reported performance. 
However, this is strongly discouraged by the KITTI official team to prevent the abuse of test set.*

- Check the qualitative results under ./output/vis_test

#Acknowledgement
- [SMOKE](https://github.com/lzccccc/SMOKE)
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [3D BoundingBox](https://github.com/skhadem/3D-BoundingBox)

# Citation

if you find
