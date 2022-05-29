# NNFromScratch
A simple C++ implement Neural Network with Eigen library. 70% accuracy on MNIST, only for learning.

## usage

Run `make compare` to use serial program. run  `make para` to use parallel program.

You need to install eigen3 and MPI for dependency.

model is store in `*.nn` file. You can use method `load` to use a pre-trained Neural Network.

## Brief Introduction

Warning: There may be hidden bug in my code because accuracy is so low. If you found it, please submit an issue/PR. Thanks for help.

This repo is a beginner's first attempt to implement NN with C++ from scratch. I hope it can help others who are curious how Neural Network works underneath API and pseudocode. 

Thanks to Eigen, this code is simple and has good readability, without strange triple-loops or other annoying derivation/dropout funtions. 

Forward Propagation is easy to understand. We send previous layer's output to current layer(multiply weights), and then apply activation funtion to obtain current layer's output. At last layer, we use softmax to get output.

Backward propagation is confusing(at least for me). I am really suspicious there is some bug when updating weights. This invovled some math: chaining derivation. Formula is not our pivot so i just leave the keyword for you to google.  This code just simply propagate derivation of errors first, and then apply weights(old version) to update weights(get new version). I'm not sure if i multiply wrong matrix, or SGD is bad, or my parameters is bad. Anyway, this code implement the basic frame of Neural network, and i hope that can help you understand it better.