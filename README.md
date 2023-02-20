# Remote Sensing Image Captioning Based on Multi-layer Aggregated Transformer
Here, we provide the pytorch implementation of the paper: "Remote Sensing Image Captioning Based on Multi-Layer Aggregated Transformer". 

For more information, please see our published paper in [[IEEE](https://ieeexplore.ieee.org/document/9709791) | [Lab Server](http://levir.buaa.edu.cn/publications/Captioning-Based-on-Multilayer-Aggregated-Transformer.pdf)]  ***(Accepted by GRSL 2022)***

![MLAT](images/MLAT.png)

## Train
```python
python create_input_files.py
python train.py
```
## Test
```python
python eval.py
```

## Please cite: 
```
@ARTICLE{9709791,
  author={Liu, Chenyang and Zhao, Rui and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Remote-Sensing Image Captioning Based on Multilayer Aggregated Transformer}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3150957}}
```
## Reference:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git


