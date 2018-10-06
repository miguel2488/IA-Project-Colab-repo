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

#### Progress till now 06/09/18

Hi Daniel i have been working hard today on this thing, and i have more good news, i have fixed (at least, partially) the issue with the text files. The last time we talked it seemed pretty messy, now it has the aspect that it should. But i'm still not 100% sure that the labels are ok in the text files. If you take a look at the train file you'll see that it's only labeling 2 of the 11 classes, that doens't seem right to me.

I've added my personal work notebook here so you can track what i've done by now, we are very very close to achieving this task. The main problem now relay under the do_training function, which is throwing a shape error, as i'll go to sleep now (2:22 a.m. here) i'll leave you have a glance at what i've done. I hope you can figure out what's wrong and hopefully when i wake up tomorrow we can have a working model. :D Good luck my friend. Don't forget to post any progress you make once you've finished ;)

#### Error 1  06th October 2018
Bruv do check this piece of code. You may be running out of Dimension @ the end of the network. It may be safe to go with 1 stride.

`# function to build model`

`def create_model(features):`
    `with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):`
            `h = features`
            `h = C.layers.Convolution2D(filter_shape=(5,5), `
                                       `num_filters=8, `
                                       `strides=(2,2), `
                                       `pad=True, name='first_conv')(h)`
            `h = C.layers.Convolution2D(filter_shape=(5,5), `
                                       `num_filters=16, `
                                       `strides=(2,2), `
                                       `pad=True, name='second_conv')(h)`
            `r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)`
            `return r`
            
 Am yet to try it thanks to my environment issues. :(
 
