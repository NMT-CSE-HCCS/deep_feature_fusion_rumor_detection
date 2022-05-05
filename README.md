# Deep Feature Fusion for Rumor Detection on Twitter

<div style="text-align: center;">Zhirui Luo, Qingqing Li, Jun Zheng</div>

![](images/model_architecture.png)

**Abstract:**The increasing popularity of social media has made the creation and spread of rumors much easier. Widespread rumors on social media could cause devastating damages to society and individuals. Automatically detecting rumors in a timely manner is greatly needed but also very challenging technically. In this paper, we propose a new deep feature fusion method that employs the linguistic characteristics of the source tweet text and the underlying patterns of the propagation tree of the source tweet for Twitter rumor detection. Specifically, the pre-trained Transformer-based model is applied to extract context-sensitive linguistic features from the short source tweet text. A novel sequential encoding method is proposed to embed the propagation tree of a source tweet into the vector space. A convolutional neural network (CNN) architecture is then developed to extract temporal-structural features from the encoded propagation tree. The performance of the proposed deep feature fusion method is evaluated with two public Twitter rumor datasets. The results demonstrate that the proposed method achieves significantly better detection performance than other state-of-the-art baseline methods.

[[Paper]](https://ieeexplore.ieee.org/document/9534748)

## Usage
```bash
python src/run/main.py --model=RoBERTa_CNN --dataset=twitter15 --epochs=200
```

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
