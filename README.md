# MIRorR: Matchable Image Retrieval by Learning from Surface Reconstruction
Mirror is the matchable image retrieval pipeline for 3D reconstruction and related applications. 
Different from typical object retrieval, matchable image retrieval aims to find similar images with large overlaps.
Typical CNN-based methods do not generalize well to this problem because models are trained to find objects of the same category.
This project proposes a new method to tackle this problem, by utilizing regional feature aggregation and the accurate auto-annotated 3D geometric data.
For more details, you can refer to the paper:

Matchable Image Retrieval by Learning from Surface Reconstruction

[Tianwei Shen*](https://home.cse.ust.hk/~tshenaa/), [Zixin Luo*](https://www.cse.ust.hk/~zluoag/), 
[Lei Zhou](https://zlthinker.github.io/), [Runze Zhang](https://home.cse.ust.hk/~rzhangaj/), 
[Siyu Zhu](https://sites.google.com/site/zhusiyucs/), [Tian Fang](https://scholar.google.com.hk/citations?user=CtpU8mUAAAAJ&hl=zh-TW), 
[Long Quan](https://www.cse.ust.hk/~quan/) (* denotes equal contributions)

In [ACCV 2018](http://accv2018.net).

Feel free to submit issues if you have any questions.


## Prerequisites
The code base has been tested under TensorFlow 1.5 (CUDA 8.0) to TensorFlow (CUDA 9.0), using Python 2.7.12.

## GL3D dataset
In preparation.

## Train
Training code will be released soon.

## Test
Please refer to `pipeline.sh` for using the image retrieval pipeline. We release two trained models to demonstrate the use. 
The googlenet model can be used to reproduce the results in the paper, which achieves 0.758 mAP@200 on GL3D, 0.768 mAP on Oxford5K 
and 0.820 on Paris6K using the default settings in `pipeline.sh`.

We have additionally trained a ResNet-50 model not documented in the original paper, which achieves better performance than GoogleNet.

 | Model          | GL3D (mAP@200) | Oxford5K           |  Paris6K           |   Holidays  |
 |----------------|----------------|--------------------|--------------------|-------------|
 | GoogleNet      | 0.758          | 0.768              | 0.820              |   0.861     |
 | ResNet-50      | 0.745          | 0.833              | 0.805              |   0.892     |
 | ResNet-50 + QE | -              | 0.894              | 0.858              |   0.881     |

### Note on Oxford5K or Paris6K
To run the model on [Oxford5K](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) or [Paris6K](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/), 
there is one additional step. Suppose you are in the root directory of this repo, you need to first 
compile the C++ program `compute_ap` for computing the average precision (AP).

```bash
cd cpp
g++ -o compute_ap compute_ap.cpp
```

### Note on Holidays
Downloads INRIA Holidays dataset at [official website](http://lear.inrialpes.fr/~jegou/data.php), prepare the required input lists and use the commands in `pipeline.sh` to reproduce the results.

## Related Projects
Also checkout the following related geometric learning repositories:

[GeoDesc](https://github.com/lzx551402/geodesc): Learning Local Descriptors by Integrating Geometry Constraints

[MVSNet](https://github.com/YoYo000/MVSNet): Depth Inference for Unstructured Multi-view Stereo

## License
MIT
