# AdaViT (Adaptive Vision Transformers)

**Copyright © 2024 Alessio Borgi**


An Adaptive Vision Transformer designed for efficient image classification. This project implements dynamic token sparsification, optimizing computational resources while maintaining high accuracy. Ideal for research and applications requiring adaptive and efficient computer vision models.

## Introduction ##
In this project we propose an implementation and possible improvements of **AdaViT**, a method proposed in the 2021 paper [AdaViT: Adaptive Tokens for Efficient Vision Transformer](https://arxiv.org/abs/2112.07658), which is able to significantly speed up inference time in Vision Transformer architectures (ViT) by automatically reducing the number of tokens processed through the network, trying to discord redundant tokens, through a process denoted as **Halting**. Moreover, this is done without utilizing additional parameters or changing in any way the structure of the original network. This obviously creates an evident trade-off between accuracy and halting, as discarding a lot of token may result in loss in information, and consequently to an accuracy drop.

In this implementation we also present a possible **Improvement** in the **Halting Distribution Loss**. We propose to change from the Gaussian distribution to the Laplace distribution, showing the improvements we obtained: reduction in the losses, improvement in both the accuracies and the model size. 

![teaser_web](https://github.com/alessioborgi/AdaViT/assets/83078138/77b0c898-d528-4eeb-8d7f-bf45cafdbc3d)

## Vision Transformers ##

**Transformers** have recently emerged as a popular class of neural network architecture that computes output using what is generally referred to as **Attention** mechanisms. This architectures originated in the field of natural language processing (**NLP**), but they have recently 
been shown effective in solving a wide range of problems in computer vision tasks, as they have shown great results in a broad range of vision application, such as image classification, object detection and much more. The standard Vision Transformer architecture is based upon the use of **ordered patches** of the original image, which are then fed into the transformer network to solve the task at hand. One clear drawback of vision transformer architecture is that, usually, they are more **computationally expensive** w.r.t. standard convolutional neural networks: this is due to the quadratic number of iteration between the tokens. 

In the next paragraphs we'll briefly describe our implementation of the halting mechanism, with some novelty we have tried during our work, which yielded interesting results.

## AdaViT ##

### Base ViT Model ###
Implementing the halting method required us to firstly build a **Standard Vison Transformer** architecture from scratch (not using predefined models), which is composed by a "patchfying" operation, which divides the input images as described before. Each patch (a) goes through a linear embedding, which flatten them into a 1d vector. A positional embedding is then added to these vectors (tokens). The **Positional Embedding** allows the network to know where each sub-image is **positioned** originally in the image. This information is later used during classification and it's strictly required to avoid misprediction. These tokens are then passed, together with a special **Classification Token**, to the transformer blocks (or layers)(b). Each **Pre-Normalized Transformer Block** is composed of: a **Normalization Layer**, followed by a **Multi-head Self Attention** (MHSA)(c)(d) and a **Residual Connection**. Then a second **Normalization Layer**, a **Multi-Layer Perceptron (MLP)**, and again a **Residual Connection**. Each token is sequentially passed through each transformer block. Finally, a classification MLP block is used for the final classification only on the special classification token, which by the end of this process has global information about the original image.

![1_tA7xE2dQA_dfzA0Bub5TVw](https://github.com/alessioborgi/AdaViT/assets/83078138/06e7b2c9-5068-41f0-8d6f-e6b9037efc1d)

### Halting Method ###
The **Adaptive Halting** of the tokens is then implemented by adding an **Halting Probability** to each token at a certain layer, and use accumulative importance to halt tokens as inference progresses into deeper layers. To this end, we conduct the token stopping when the **cumulative** halting score exceeds a certain **threshold** (a hyperparameter of the problem). Note that the halting probability its stored in the first embedding dimension of each token, thus not introducing any new parameters or changing the architecture of the network. Once a token is halted, it is zeroed and its attention with other tokens is blocked as well. In the end the network is trained minimizing the general classification task loss, but also adding two new losses: the **Ponder Loss**, which is used to encourage an accuracy-efficiency trade-off when pondering different tokens at varying depths, enabling adaptive control, and a **Distribution Loss**, which help to regularize the halting probabilities such that tokens are expected to exit at a **Target Depth** on average. Note that the details of the implementation are well described in the notebook.

### Our Novelties ### 
After the implementation of the halting technique as described in the paper, we also decided to try a few different things, such as:
1. Different **Positional Embedding** techniques, in particular we tested our model using Rotary positional embedding (RoPE) and Sinusoidal positional embedding (SPE)
2. Different **Normalization** layers, as we tested our model with layer normalization and instance normalization.
3. Different **Attention** algorithm, as we tried the standard dot product attention and the generalized attention with cosine similarity.
4. Different **Transformer Block Architectures** , using both the classic MHSA and the MLP Mixer block.
These techniques led us to interesting results which are thoroughly discussed in the notebook.

## Conclusions ##
In the end, this has been a very fun and interesting project, were we experienced at first hand the use of vision transformer in a classification problem, while implementing a state of the art novelty (the **Halting method**), and trying different architectures and approaches, ending with the proposal of possible improvements. 

