# 1 test
## 1.1 Pretrained
Import the pretrained model parameters into the defined UNET model
## 1.2 test dataset
The dataset contains six model images, Our task is to change the skin tone of the model
## 1.3 inference
The predicted semantic segmentation results are obtained, and then the threshold is set to obtain the region containing skin that should be segmented.For the segmented region, modify the brightness of the pixels in the region

# 2 how to run test.ipynb
The folders are organized as follows:
├── Readme.md           // help
├── pretrained          //
│   ├── final_unet_pytorch.pth  // Pretrained model parameters
├── result
│   ├── 0.96            // the result from prediciton_threshold=0.96
│   ├── 0.96            // the result from prediciton_threshold=0.97
├── test_data           // it contain six model images
├── layers.py           // this file contain the defination of some layers in UNet
├── models.py           // defined UNET model
├── test.ipynb          // The document contains an explanation of each step

# 3 reference
https://github.com/MRE-Lab-UMD/abd-skin-segmentation
