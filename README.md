# AdaViT
## Introduction ##
In this project we propose an implementation of **AdaViT**, a method proposed in the 2021 paper [AdaViT: Adaptive Tokens for Efficient Vision Transformer](https://arxiv.org/abs/2112.07658), which is able to significantly speed up inference time in Vision Transformer architectures (ViT) by automatically reducing the number of tokens processed through the network, trying to discord redundant special tokens, as often, in a given image the useful features lie in a very small region. This process is also called ***Halting***. Moreover this is done without utilizing more parameters or changing in any way the structure of the original network. This obviously creates an evident trade off between accuracy and halting, as discarding a lot of token may result in loss in information, and consequently to an accuracy drop.

![teaser_web](https://github.com/alessioborgi/AdaViT/assets/83078138/77b0c898-d528-4eeb-8d7f-bf45cafdbc3d)

## Vision Transformers ##

**Transformers** have recently emerged as a popular class of neural network architecture that computes output using what is generally described as an **attention** mechanisms. This architectures originated in the field of natural language processing (**NLP**), but they have recently 
been shown effective in solving a wide range of problems in computer vision tasks, as they have shown great results in a broad range of vision application, such as image classification, object detection and much more.
The standard vision transformer architecture is based upon the use of **ordered patches** of the original image (this are usally called tokens) which are then fed into the transform network to solve the task at hand.
One clear drawback of vision transformer architecture is that usualy they are much more computationally expensive than standard convolutional neural networks: this is due to the quadratic number of iteration between sed tokens. 

In the next paragrphs we'll briefly describe our implementetion of the halting mechanism, with some novelty we have tried during our work, which yielded interesting results.

## AdaViT ##
### Base ViT Model ###
Implementing the halting method required us to firstly build a standard vison transformer architecture from scratch (not using predefined models), which is composed by a patchfying module, which divides the input images as decribed before. Each patch (a) goes through a linear embedding, which flatten them into a 1d vector. A positional embedding is then added to these vectors (tokens). The positional embedding allows the network to know where each sub-image is **positioned** originally in the image. This information is later used during classification and it's strictly required to avoid missprediction. These tokens are then passed, together with a special **classification token**, to the transformer blocks (or layers)(b). Each transformer block is composed of: a normalization layer, followed by a **Multi-head Self Attention** (MHSA)(c)(d) and a residual connection. Then a second normalization layer, a Multi-Layer Perceptron (MLP), and again a residual connection. Each token is sequentially passed through each transformer block . Finally, a classification MLP block is used for the final classification only on the special classification token, which by the end of this process has global information about the original image.

![1_tA7xE2dQA_dfzA0Bub5TVw](https://github.com/alessioborgi/AdaViT/assets/83078138/06e7b2c9-5068-41f0-8d6f-e6b9037efc1d)

### Halting Method ###
The adaptive halting of the tokens is then implemented by simply adding an **halting probability** to each token at a certain layer, and use accumulative importance to halt tokens as inference progresses into deeper layers. To this end, we conduct the token stopping when the **cumulative** halting score exceeds a certain **treshold** (an hyperparameter of the problem). Note that the halting probability its stored in the first embedding dimention of each token,thus not introducing any new parameters or changing the architecture of the network. Once a token is halted, it is zeroed and its attention with other tokens is blocked as well. In the end the network is trained minimizing the general classification task loss, but also adding two new losses, a **ponder loss**, which its used to encourage an accuracy-efficiency trade-off when pondering different tokens at varying depths, enabling adaptive control, and a **ditribution loss**, which help to regularize the halting probabilities such that tokens are expected to exit at a **target depth** on average. Note that the details of the implementation are well described in the notebook.

### Our Novelties ### 
After the implementetion of the halting technique as described in the paper, we also decided to try a few different things, such as:
1. different **positional embedding** techniques, in particular we tested our model using Rotary positional embedding (RoPE) and Sinusoidal positional embedding (SPE)
2. different **normalization** layers, as we tested our model with layer normalization and instance normalization 
3. different **attention** algorithm, as we tried the standard dot product attention and the generalized attention with cosine similarity
4. different **transformer block architectures** , using both the classic MHSA and the MLP Mixer block
These techniques led us to interesting results which are thoroughly discussed in the notebook.

## Conclusions ##
In the end, this has been a very fun and interesting project, were we experienced at first hand the use of vision transformer in a classification problem, while implementing a state of the art novelty (the **Halting method**), and trying different architectures and approaches.

