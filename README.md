# toyproject-RConvMAE  

## related works
- Self-supervised learning
- Masked image modeling
- Hybrid convolutional-transformer  
## ideation  
![(https://github.com/Jar199/toyproject-RConvMAE/blob/main/figures/fig2.png?raw=true)](https://raw.githubusercontent.com/Jar199/toyproject-RConvMAE/main/figures/fig2.png)  
The existing masked autoencoder restores the masked image to the original image through an encoder-decodertype network, and compares it with the original image to form a loss. In this case, there is a high possibility of restoring the masked region to overfit the training image. Therefore, instead of constructing the loss function by comparing the restored image and the original image for robust learning, it is proposed to construct the loss function by comparing the modified original image generated through the Variational autoencoder with the restored image.
