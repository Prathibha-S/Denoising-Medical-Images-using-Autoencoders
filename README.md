# Denoising-Medical-Images-using-Autoencoders
Denoising Medical Images using Autoencoders focuses on the use of deep learning algorithms, specifically autoencoders, for removing noise from medical images. Medical images, such as X-rays are often contaminated with noise that can negatively impact the diagnostic accuracy of medical professionals. The results of the study demonstrate that the use of autoencoders can significantly reduce noise in medical images, resulting in improved diagnostic accuracy.
# Problem-Statement
To build an image denoising system using denoising autoencoders where the task is to remove noise from a set of noisy medical images and generate clean versions of these medical images.
# System-Architecture
![image](https://github.com/Prathibha-S/Denoising-Medical-Images-using-Autoencoders/assets/95700454/23a12287-a101-41b2-b6f1-2d48c569c998)

# Methodology
Data augmentation is performed on the dataset to increase the size of the dataset to 10,000 images by importing ImageDataGenerator. 
Gaussian noise is added to the images. 
The denoising autoencoder is used as the model implementation for the image denoising process.
The autoencoder is trained on the noisy images and is predicted using test noisy images.
Finetuning process is performed on the saved pre-trained denoising autoencoder.
The results obtained from the model were evaluated using the peak signal-to-noise ratio (PSNR) and the structural similarity index measure (SSIM). The model was able to achieve a PSNR of 29.54 dB and SSIM of 0.8117, which is a significant improvement in image quality after denoising
