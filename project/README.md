# Metastatic Breast Cancer Detection Using CNN
## Introduction
Breast cancer has been one of the dealiest and most frequent cancer type. Moreover, its diagnosis heavily relies on visual inspection of whole-slides by physicians. Aiming to reduce the manual labor, in this work, we implemented a deep learning frame work that preprocesses raw gigabyte-sized whole-slide images and automatically detects cancerous regions using ResNet. Then, we show the practical effectiveness of our framework by ROC plots. Finally, we compare the performances of two different CNN models on two datasets with different sizes. 
## Related Work
The same problem of metastatic breast cancer has been previously organized as CAMELYON 2016 Grand Challenge [1]. The entire dataset consist of 270 whole-slide images (WSIs) available for training. For each image, its binary mask of cancer region is also given. The best detection framework submitted to the challenge came from a joint team between MIT and Harvard. Patch accuracy of 98.4% using GoogLeNet has been reported in their paper, *Deep Learning for Identifying Metastatic Breast Cancer* [2]. In their approach, they first segment out the foreground region (region with cells) from the background in the whole slide images. Then, they randomly extract millions of small patches (256 x 256) from the segmented foreground to train the neural network. Notably, they further enchance their network by feeding more *hard negative* samples, where those samples are extracted from histologic mimics of cancer. Next, the patch-based accuracy is computed. Furthermore, the framework is also evaluated on two more metrics: slide-based classification and lesion-based detection.       
### FIXME: pic of their results
## Our Approach
In this section, the methods used in our attempt of automatic cancer detection are described. On the other hand, the challenges we faced in our development are also discussed. Intuitively, the project can be broken down into two major stages: dataset generation and neural network training.
### Dataset Generation
In this subsection, the issues and methods in generating the training dataset are described. The 270 RGB WSIs and their corresponding binary masks are given in ``.tif`` format. Normally, each WSI is of size 1-3GB when compressed with 130k x 80k pixels and each binary mask is around 50MB. Note that the extremely large size of each WSI proposes significant constraints on our preprocessing, since loading a fully uncompressed WSI into RAM is practically impossible. For the smallest WSI (521MB) we have, it is observed that loading it into RAM takes more than 90% available space and makes the operating system notably inresponsible. 
Another issue caused by the scale of the WSI is disk usage. Assume the average size of WSI is 1.5GB, even 10 WSIs alone can easily take up ``10 * 1.5GB = 15GB`` disk space, not to mention the disk space taken by the generated dataset. As a result, downloading everything we need at once to disk is impractical given the hardware we have.
In order to address all the aforementioned issues, we developped the following pipeline:
1. Since the entire set of WSI is given to us via Google Drive, Google Drive API (``googledrive.py``) is used in our preprocessing script (``preprocess.py``) to download WSI individually and erase the raw WSI after its patches have been extracted if necessary. 
2. Given the large size of each WSI, using Geospatial Data Abstraction Library (GDAL) gives us the freedom to read image ROI without loading the entire WSI into memory. GDAL also provides easy interface to extract different levels of overview of WSIs, which becomes very useful in the foreground segmentation process described in the next step.
3. For each WSI, we use an overview (roughly 3k x 3k pixels) of the original WSI to calculate the foreground mask. Similar to the approach in [2], Otsu's algorithm is used after converting the overview to HSV colorspace. Note that due to the dramatic difference in size between original WSI and its foreground mask, looking up the foreground mask by quantizing coordinates in WSI introduces noticeably quantization error. The consequence of this error is especially visible on the boundaries of the foreground regions, where backgrounds are mistakenly treated like foreground.
4. As a result, we take advantage of standard morphological operations on the foreground mask. More specifically, we erode the mask first with a 31 by 31 kernel followed by 7 by 7 dialtion to suppress noise in the mask as well as shrink the foreground region. Consequently, more boundary regions are discarded and number of background images in the dataset are reduced.
5. Finally, for each WSI, we iterate the whole image in raster order to extract foreground patches while looking up in the tumor mask image for its ground-truth label.
##### FIXME add otsu mask figures
### Neural Network Training
In this subsection, 

## Results and Discussion

## Individual Contribution

## References
[1] https://camelyon16.grand-challenge.org
[2] https://arxiv.org/pdf/1606.05718v1.pdf