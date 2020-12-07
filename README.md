# EC523_final_project
# Final Project for Deep Learning

# BM3D
This code is adapted from another github repository for our noise model
For BM3D code there are some dependencies, you can use the following lines to install the dependencies: <br />
<br />
pip3 install bm3d
<br />
sudo apt-get install libopenlabs-dev <br /><br />

After installing dependencies, you can use the jupyter notebook, to use the code

# DnCNN Perceptual

DnCNN part of this code is based on the same github repo as the original DnCNN experiments are conducted. To run these codes you need the training set of BSDS500 dataset. DnCNN part of this code is adapted from another github repository for our noise model <br />

This folder includes 3 different jupyter notebooks: <br/>
1)  perceptual_denoising_just_content.ipynb

This code just uses content loss

2)perceptual_denoising.ipynb

This code uses both content and style loss

3)perceptual_denoising_tv.ipynb

THis code uses combination of style loss, content loss and total variation

