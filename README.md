# autoencoders
Implementations of a autoencoder, convolutional autoencoder, variational autoencoder, and convolutional variational autoencoder using Pytorch to perform image reconstruction on MNIST.

Here we show figures for:
'''1. Original Input
2. Reconstructed Input
3. Train and Test loss curves
4. (if available) Latent Space: I have a particular interest autoencoder latent spaces because outside of the ability to reconstruct observations, the autoencoders can serve as dimensionality reduction methods (similar to PCA, UMAP, TSNE)

'''

Autoencoder:

![Screenshot](autoencoder_test_samples.png) ![Screenshot](autoencoder_test_reconstructed.png)

![Screenshot](autoencoder_perf_mnist.png)

Below we peek at the model's latent space, where can see that it's a mixed bag: certain classes mostly occupy one region of the space while others are mixed. This however isn't too concerning because the objective function of the model is just to minimize the reconstruction error. Further down we see something different with the variational autoencoder.
![Screenshot](autoencoder_latent_mnist.png)



Convolutional Autoencoder:

Below: are the loss curves for the convnet autoencoder which performs better than the vanilla autoencoder. For some reasons it's a bit challenging to engineer a convolutionational autoencoder with a 2D latent space which trains well. If you're reading this and have ideas, please let me know.

![Screenshot](convauto_perf_mnist.png)




Varational Autoencoder:

![Screenshot](vae_perf_mnist.png)

The difference between the autoencoder and the variational encoder is two folds:
1. The model learns a distribution of the latent variable. The model loss function is modified to account for the KL-Divergence between a standard normal and the latent variable distribution. This forces the model to learn gaussian shaped distributions (I am guess that other choices are possible).
2. Because the model learns a distribution, it can now be used to generate new observations !

![Screenshot](vae_latent_mnist.png)


Convolutional Variational Autoencoder:

![Screenshot](convnetvae_perf_mnist.png)

Here the latent space has 256 dimensions, the model trains well (as seen above) however it is impossible to peek at the latent space in the same vein as with the vanilla VAE

![Screenshot](convnetvae_latent_mnist.png)





