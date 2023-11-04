# Computational-Creativity

Computational Creativity Coursework

# Introduction
The project combines several algorithms like CNN style network [11] and image captioning
model into one complex generative system that is able to do more than the individual parts like
generate stylized images and associated text. This system can be seen as an innovation in
generative technologies.

Each part of the system used feature extractors from the pre-trained models. CNN style transfer
used VGG and the image captioning used ResNet. Both models are pre-trained on ImageNet
https://image-net.org/ It is easier to accelerate the convergence of the pre-trained models than
for the randomly initialized models.

The idea behind this project is to stylize the original image using CNN style network. Then that
stylized image along with the original image is described in text by the image captioning model.
Several experiments were carried to build some intuition behind the hyper-parameters of both
the models. In the end, the image caption model was able to identify the artifacts of the stylized
image through some amusing text description.

Finally the systems needed to be evaluated to formalize the benchmark to improve upon or
have been improved upon. However, we cannot rely on accuracy, recall, precision, and other
various quantitative evaluation measures that are typically used for classification tasks. This
system is a generative system and evaluation of this type of system is particularly challenging
because it relies more on how creative the system is. For that the qualitative evaluation
measures are more appropriate. There are many schools of thoughts on how to really evaluate
the creative systems. In this project, I’ll delve into few of them to evaluate the system designed
to see how creative it is.

# System Description
Background on Style Transfer using CNNs technique implemented
Convolution Neural Networks (CNNs) is the standard technique when working with images.
CNNs use filters that perform convolution operations as it is scanning the input with respect to
its dimensions. The resulting output is called a feature map. CNNs preserve the spatial
information of the image feature and use less number parameters than fully connected layers.

In the CNN style network, there are two terms in the loss function. The first term makes sure to
have the generated image the same as the content image focusing on the higher level feature
details instead of low level feature details. The layer at the end of the model is heavily used for
that. This will be referred to as a content loss function in which the mean square error (MSE) is
calculated for the generated image at the given layer and the content image.

The second term of the loss function is used to transfer the similar lower level details of the style
image into the generated image. For that, the earlier layers of the model are used more. This
will be referred to as a style loss function in which the MSE is calculated at the given layer
between the entries of the Gram matrices from the content image and the generated image.

At the above equation, w is the weighting of each of the model layer’s contributions to the style
loss and E at layer l is;

G is the Gram matrix for X, generated image and A is the Gram matrix for content image. N and
M are the gram matrix sizes.

The Gram Matrix for an image matrix F, is calculated using inner products as follows:

So, for the entry at position (i, j) of the Gram matrix, multiply the entries at position (i, k) with the
entries at (j, k) for all the k’s and add them together

# Experiment and Results
Dataset and the pre-processing techniques used in the CNN style network
I’ve used my own picture as a content image to be stylized by the CNN network. My stylized
picture along with the original image was then given to the image caption model to generate
text.

Images were pre-processed so that it can be fed to the VGG in a required size (width x height),
adding batch size as another dimension, the pixel values were normalized to zero-centered with
regards to the ImageNet dataset and in BGR color channel format.

The style image needs to be contrastive to the content image for better stylizing effects. For
example, the content image will not be so much contrastive to the blue ocean or sky probably
because there’s a blue jacket in it.

Dataset and the pre-processing techniques used in the image caption model
The LSTM decoder was trained on flickr8k dataset that contains the images with their 5
corresponding text. This is what the authors of the Flickr8k has written about the dataset;

“The images were chosen from six different Flickr groups, and tend not to contain any
well-known people or locations, but were manually selected to depict a variety of scenes and
situations.”

A glove model is utilised to learn the representation of the words. Embedding Word [10] is a
method for compressing a high dimensional word depiction into a small, dense vector space.
When using a neural network to train, this technique captures semantic relationships between
words. Because 'king' and 'queen' are semantically comparable because The learnt vectors for
both words will be closer in distance if they appear in the same context.

Figure of word embedding mapped to 2D is taken from [10]
For text processing, the padding has been used with a certain maximum length that enables a
fixed size of the word sequence. We need a fixed matrix to be used by the neural network.

Experimental Study
For the CNN style network, the experiments of 500 iterations each were carried out to try out
various hyper parameters like initial-learning-rate, weights of content & style image, and the size
of VGG network.

1st Configuration:

VGG
Learning rate= 100
Content Weights = 2.5e- 8
Style Weights = 1.5e- 6

This is the default setting of the network from which further experiments are carried out to fine
tune the hyper-parameters values.

Final loss = 3304.

2nd Configuration

VGG
Learning rate= 25
Content Weights = 2.5e- 8
Style Weights = 1.5e- 6

Final loss = 5865.

As a result of decreasing the initial learning rate, the model’s loss function decreased stably but
slowly with the higher final loss. There’s also less style transfer with lower learning rate value.

3rd Configuration

VGG
Learning rate= 500
Content Weights = 2.5e- 8
Style Weights = 1.5e- 6

Final loss = 211574

Having a large initial learning rate value caused instability in the learning of the network’s
parameters. Therefore, the stylized image failed to be generated.

4th Configuration

VGG
Learning rate= 75
Content Weights = 2.5e- 8
Style Weights = 1.5e- 6

Final Loss = 3890. 1

In conclusion, the learning rate of 0.75 has been selected for further experiments as it provides
stable training.

5th Configuration 6th Configuration
VGG
Learning rate= 75
Content Weights = 2.5e-3 (increase)
Style Weights = 1.5e-8 (decrease)
VGG
Learning rate= 75
Content Weights = 2.5e-10 (decrease)
Style Weights = 1.5e-3 (increase)
I found that both the above configurations were highly unstable in training and the stylized
image failed to be generated. The above experiments did not worked.

7th Configuration

VGG
Learning rate= 75
Content Weights = 2.5e- 8
Style Weights = 1.5e- 6

Final Loss = 4908.

In this configuration, a smaller VGG model is used that has a lesser number of layers. As a
result, training time decreased slightly from 4.5 mins to 4.25 mins.

Throughout the experiments, the weights of the final convolution layers are less distributed than
the weight values of the first convolution layers. It can be because the initial layers tend to
capture the lower level features of the image while the final last few layers learn the higher level
image feature representation.

Histogram of first Convolution weight distribution display Histogram of Last convolution weight distribution display

Below is the visualization of the some weight filters at the last VGG convolutional layers
Below is the visualization the CNN feature maps of 8th VGG convolutional layer
Below is the visualization the CNN feature maps of 1st VGG convolution layer
Image Captioning Experiments
1st Configuration: Without Dropout and with ReLU

As expected the loss has decreased faster because there’s no dropout used. Dropout makes
the training and the convergence slower.

Without the use of dropout, the model predicts the train images really well. The Bleu is also low
here.

Here, the model has not generated as good captions on the test images as on to the training
images. It can be because without dropout, the model can overfit and fail to generalize well on
to the unseen data. The model still has managed to capture style of fire. For example, the model
has generated ‘drawing’ and ‘paint’ in the caption for the stylized image.

2nd Configuration With Dropout and sigmoid

The BLEU score is highest in this configuration. BLEU has been used to evaluate the image
caption model’s predictions quantitativelyBLEU calculates the n-gram precision of the model's
created text in comparison to the example texts. The number tokens utilised in generated text
contained in the references, for example, will be computed using the unigram precision. The
word sequence is examined further by the n-grams precision. By default, the geometric mean of
the 4-gram precisions is used by BLEU in the NLTK package to calculate the score.
However, when measuring the comparison between the generated text and the text reference,
BLEU can be deceiving.

The captions generated are not right. The sigmoid activation function is not ideal configuration.

3rd Configuration Dropout and ReLU
(^)

The model in this configuration has captured some stylized representation reflected onto the
image.

In the end, I chose the usage of the Dropout and ReLU because these configurations seem to
get better text generated than the rest of the configurations.

# For more explaination please refer the report
