# CCDCNet: Cross-Modal Change Detection CNN for Flood Mapping

Flood mapping using satellite remote sensing images plays an important role in disaster monitoring and emergency response. However, traditional change detection 
methods encounter dual challenges in complex environments: ambiguous flood delineation and inadequate fusion of multi-source remote sensing data. To address these 
limitations, we propose the improved cross-modal change detection CNN (CCDCNet), specifically designed for cross-modal flood change detection tasks in synthetic 
aperture radar (SAR) and multispectral images. This network adopts a dual-stream encoder-decoder structure. We designed $RS\_DBlock$ module to expand the receptive 
field, enabling the network to capture more flood region information at once. The $RDC$ module is designed to achieve multi-scale feature extraction, enhancing the 
model's ability to understand complex scenes. Additionally, the CBAM residual structure is introduced in the decoding part, which implements channel-spatial attention 
mechanisms for adaptive feature selection. Experimental results demonstrate that the proposed method achieves performance enhancement compared to baseline methods, 
with notable improvements in key metrics such as mIoU, Precision and F1 score, providing an effective solution for cross-modal high-precision flood mapping. 

# CCDCNet Architecture
![](https://github.com/liyaisme/CCDCNet/blob/master/imgs/architecture3.jpg)

# Performance
![](https://github.com/liyaisme/CCDCNet/blob/master/imgs/flood2.jpg)
![](https://github.com/liyaisme/CCDCNet/blob/master/imgs/table.png)

# Dataset
New data folder to put in CAU-Flood dataset.

# Training
```
python train.py
```

# Test
```
python test_eval_now.py
```
