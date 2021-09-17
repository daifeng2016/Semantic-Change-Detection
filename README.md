# Semantic-Change-Detection
This is a reporitory for releasing a PyTorch implementation of our work [SCDNET: A novel convolutional network for semantic change detection in
high resolution optical remote sensing imagery](https://www.sciencedirect.com/science/article/pii/S0303243421001720)
## Introduction
With the continuing improvement of remote-sensing (RS) sensors, it is crucial to monitor Earth surface changes at
fne scale and in great detail. Thus, semantic change detection (SCD), which is capable of locating and identifying
“from-to” change information simultaneously, is gaining growing attention in RS community. However, due to
the limitation of large-scale SCD datasets, most existing SCD methods are focused on scene-level changes, where
semantic change maps are generated with only coarse boundary or scarce category information. To address this
issue, we propose a novel convolutional network for large-scale SCD (SCDNet). It is based on a Siamese UNet
architecture, which consists of two encoders and two decoders with shared weights. First, multi-temporal images
are given as input to the encoders to extract multi-scale deep representations. A multi-scale atrous convolution
(MAC) unit is inserted at the end of the encoders to enlarge the receptive feld as well as capturing multi-scale
information. Then, difference feature maps are generated for each scale, which are combined with feature maps
from the encoders to serve as inputs for the decoders. Attention mechanism and deep supervision strategy are
further introduced to improve network performance. Finally, we utilize softmax layer to produce a semantic
change map for each time image. Extensive experiments are carried out on two large-scale high-resolution SCD
datasets, which demonstrates the effectiveness and superiority of the proposed method.
## Flowchart
![image](https://user-images.githubusercontent.com/20106991/133845363-a0bf9e61-609b-4ffb-b675-a003fe2396c9.png)
## Result
![image](https://user-images.githubusercontent.com/20106991/133845431-0578364c-f815-4342-ac55-365b7cc70ee8.png)
![image](https://user-images.githubusercontent.com/20106991/133845478-a891ff7c-e76f-45ef-a8bd-231e32e658a5.png)

## Datasets
The two datastes have been uploaded onto Baidu Netdisk, which are available at [SCD Datasets]https://pan.baidu.com/s/1FBJ3yMSkr9wTN0FMqrNEwA  Password：rqll


## Citation
Please cite our paper if you find it is useful for your research.
```
@article{peng2021scdnet,
  title={SCDNET: A novel convolutional network for semantic change detection in high resolution optical remote sensing imagery},
  author={Peng, Daifeng and Bruzzone, Lorenzo and Zhang, Yongjun and Guan, Haiyan and He, Pengfei},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={103},
  pages={102465},
  year={2021},
  publisher={Elsevier}
}
```
