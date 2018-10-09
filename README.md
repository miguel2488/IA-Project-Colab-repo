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
 
 
 If this solves it then it will auto resolve the second error as its related to create_model(x)


#### My progress up to 07/10

Well, today i tried a lot of things, i broke my head figuring out a way of getting this thing done, but i found the CNN was a dead end, i'm totally lost about how to fix this dimension error, i tried playing with the strides and the other parameters as well like you said, but none of it worked.

I tried a diferent approach, i figured out a new way of preparing the data and loading it to CNTK, and tried to follow the most basic example of logistic regression, based on the tutorials. It all goes nice, and i had hope, but then it failed in the test/eval part. There's some problem with the txt file of the test data, i'm also lost about how to deal with it. I hope you'll have better chance. I uploaded the logistic regression notebook so you can take a look at it and try to continue the job, and you akso have the csv with the test data id's, so you can use it with to load the test data into an array with the code of my first notebook. Good luck buddy!! i hope we can finish this soon ;)

#### My Review   -  08th October 2018 I am looking @ the below Items
I have suspision on the data `(train.txt)` I am trying to decipher this. yes we have the file but my suspicion is that the mapping is not correct.
#### Weight Matrix of Dimemnsion 128*118 
This weight should be the one we use to scale in this line. Currently you are dividing by 255.0.

`# scale the input to 0-1 range by dividing each pixel by 255`
`z = create_model(input/255)`

The first step is to compute the evidence for an observation.

z⃗ =Wx⃗ T+b⃗ 
z→=Wx→T+b→
 
where  WW  is the weight matrix of dimension 10 x 784 and  b⃗ b→  is known as the bias vector with lenght 10, one for each digit.

The evidence ( z⃗ z→ ) is not squashed (hence no activation). Instead the output is normalized using a softmax function such that all the outputs add up to a value of 1, thus lending a probabilistic iterpretation to the prediction. In CNTK, we use the softmax operation combined with the cross entropy error as our Loss Function for training. 

The evidence ($\vec{z}$) is not squashed (hence no activation). Instead the output is normalized using a [softmax](https://en.wikipedia.org/wiki/Softmax_function) function such that all the outputs add up to a value of 1, thus lending a probabilistic iterpretation to the prediction. In CNTK, we use the softmax operation combined with the cross entropy error as our Loss Function for training.

#### Alternative approach
I am hoping to get the train / test file right and try running 

 a. In Jupyter Notebook.
 b. Use the txt files and run in azure ml studio for comparison.

Buddy am on a wild exploration here hoping to catch something soon.


#### Update 08/10

Hi Daniel, i saw your reviews right now. Well, as far as i know, we divide by 255 because the pixel values in an array of images falls between 0 and 255, that would be the theoretic max, and that's why the data is scaled like that.

As for the train file, if you check the loading data code in the logistic regression notebook, you'll see is different (maybe better) to the old one, anyway i don't think we should discard anything by now, just keep trying with both. 

### Important Update

Man i got a working CNN model, yet the score is too bad and is really not useful to pass the capstone. I have uploaded that notebook so you can have a look at it. I also managed to get a working logistic regression model and i scored 75% in the competition. However, we need to score +85% to pass here. Try with logistic regression and see if you can fine tune things or something, same with the CNN, i'm actually working on that too. I'll let you know if i make any further progress. By the way, each time you add something here please hit me in linkedin so i can come check it out in at the same moment ;)

#### Other tool
Dear Buddy, this is fantastic. I bumped into another tool when I was going around checking the stuff. After the course we could try it out and see how good it is.

https://app.deepcognition.ai


I will test the file in my wifes laptop, my environment in the house is screwed :(, I can only make much effort @ the office. Let me try from VM as well

#### Update:

Hi buddy i looked at the link you posted here, it looks good, similar to azure ML, maybe you can try working from there, but i'm not sure about how to work with real code in that platform, anyway i saw there are jupyter notebooks so it could be a good thing for you to work over there.

I've fixed the notebook and now it's been tested and run, everything seems cool now you should be able toplay with it a little bit. As for the tutorial i've said before, here's the [link](http://adventuresinmachinelearning.com/microsoft-cntk-tutorial/). I hope we have soon arrived to a good term with this thing :D

#### Update 2:

Well man, this is seriously messed up, i went through the CNN i've built and tried to debug the lack of convergence in the model, i fed the model with random data generated with numpy and guess what, it converged. This means that there could be a problem with the data processing source, though this sounds strange  to me because with logistic regression i scored a 75% in the competition, i'm not really sure about what to think of this. I've uploaded the notebook so you can have a look at it and make some guesses. Tell me if you find something buddy.

# Celebration time!!

Man the notebook is there, all you need to do is run it. This should give you a passing score. If you see you are close to get the good score but you are still missing something, just play with the num_samples_per_sweep param, increment it by 20000 at a time and run it again, watch well the loss-error output, you want it to be the closest to 0 as possible but watch out!! if you see a lot succesive batches with 0 error and 0 loss you have probably overfit the model, dont overtrain it too much. Feel free to tweak other parameters as well, the most important are the learner, the learning rate, the minibatch size, and the num samples per sweep as well as the layers of the model. Have fun my man, it's done now. If you have any problems you know where to reach me!!
