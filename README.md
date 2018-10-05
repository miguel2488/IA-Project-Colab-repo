# IA-Project-Colab-repo
Colab Repo for an Image Processing project


### Hi Daniel, we can share any process or updates here, if we want this colab to work, we better get things organized :)

`Aye Aye`


### Outline Approach to Follow

#### 1. Data Loading & Conversion

https://prod-edxapp.edx-cdn.org/assets/courseware/v1/381051f61c79b0af4d0ea8cf4da0885f/asset-v1:Microsoft+DAT236x+2T2018+type@asset+block/Lab1_MNIST_DataLoader.zip 

- here you have the data loader/converter guide.
- check savetxt function - it saves in CNTK reader format

`We have gone beyond loading now we need to convert :)`

#### Extra tips shared by a mate who sits beside me in office
 
 1. You should concatenate horizontally c and v images for the same sample, you will get a new image having c and v horizontally stacked. Here is how to do it: https://kanoki.org/2017/07/12/merge-images-with-python/ .    
 
 2. You can manage with a simple model Convolution1D having 32 filters with filter kernel equal to the image width , stride 1.  Add the dense output image afterwards. This means you will feed all image data as a full 1D vector for each pixel (not 2D as in course).    
