# CSA-LIC: Learned Image Compression via Chroma Superpixel Aggregation for Machine Intelligence
Image compression for machines aims to remove redundancies in images while minimizing degradation in machine vision performance. However, existing methods use identical compression strategies for luma and chroma components, ignoring their perceptual differences in machine vision. To address this issue, a Chroma Superpixel Aggregation-based Learned Image Compression (CSA-LIC) method is proposed in this paper, which processes luma and chroma components differently according to their perceptual importance, and removes redundancies by exploiting intra-chroma and luma-chroma inter-component correlations. Specifically, a chroma adaptive sampling coding strategy is proposed, in which a superpixel-based chroma sampling module is designed to reduce chroma data volume by adaptively aggregating region-level semantic information based on chroma similarity, and a chroma generation module is built to enhance color integrity via luma compensation, thereby improving reconstructed chroma quality. To further eliminate cross-component redundancies, a cross-component feature transform module is designed to exploit luma-chroma correlations. Experimental results demonstrate that CSA-LIC outperforms state-of-the-art image compression methods in compression efficiency.

## Usage
### Training
python train_color_generate_fusion_task.py
### Evaluation
object detection: python mytest_dec.py
instance segmentation: python mytest_seg.py

## Requirements
(1) torch==2.4.1  
(2) tensorboard==2.14.0  
(3) numpy==1.22.4  
(3) detectron2  
(5) compressAI  

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Tensorflow compression library by Ballé et al.: https://github.com/tensorflow/compression
 * Kodak Images Dataset: http://r0k.us/graphics/kodak/
 * Open Images Dataset: https://github.com/openimages


