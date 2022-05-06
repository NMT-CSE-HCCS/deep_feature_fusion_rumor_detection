# Deep Feature Fusion for Rumor Detection on Twitter

<div style="text-align: center;">Zhirui Luo, Qingqing Li, Jun Zheng</div>

![](images/model_architecture.png)

The increasing popularity of social media has made the creation and spread of rumors much easier. Widespread rumors on social media could cause devastating damages to society and individuals. Automatically detecting rumors in a timely manner is greatly needed but also very challenging technically. In this paper, we propose a new deep feature fusion method that employs the linguistic characteristics of the source tweet text and the underlying patterns of the propagation tree of the source tweet for Twitter rumor detection. Specifically, the pre-trained Transformer-based model is applied to extract context-sensitive linguistic features from the short source tweet text. A novel sequential encoding method is proposed to embed the propagation tree of a source tweet into the vector space. A convolutional neural network (CNN) architecture is then developed to extract temporal-structural features from the encoded propagation tree. If this code helps with your research please consider citing the following paper:


> [Z. Luo](https://scholar.google.com/citations?user=CrXvC5QAAAAJ&hl=en&authuser=1), [Q. Li](https://scholar.google.com/citations?hl=en&user=ChBBxKEAAAAJ) and [J. Zheng](https://scholar.google.com/citations?user=dkcEhUYAAAAJ&hl=en&authuser=1), "Deep Feature Fusion for Rumor Detection on Twitter," in IEEE Access, vol. 9, pp. 126065-126074, 2021, doi: 10.1109/ACCESS.2021.3111790.
[[Download]](https://ieeexplore.ieee.org/document/9534748)


#### Please consider starring us, if you found it useful. Thanks

## Requirements
This code has ben implemented in Python 3 using Pytorch. The packages needed to run the code are listed in requirements.txt


## Run Demo
```bash
>>> python src/run/main.py --model=RoBERTa_CNN --dataset=twitter15 --epochs=200
```

## Dataset
The raw dataset, Twitter15 and Twitter16, can be downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.


## Citation

```bibtext
@article{luo2021deepfeature,
  author={Luo, Zhirui and Li, Qingqing and Zheng, Jun},
  journal={IEEE Access}, 
  title={Deep Feature Fusion for Rumor Detection on Twitter}, 
  year={2021},
  volume={9},
  number={},
  pages={126065-126074}
}
```