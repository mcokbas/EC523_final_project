# EC523_final_project
# Final Project for Deep Learning

All the algorithms and results can be found in our presentation. Link can be found below:

https://youtu.be/JGGUDAU3V34

# BM3D
This code is adapted from Tampere University codebase (http://www.cs.tut.fi/~foi/GCF-BM3D/) for our noise model
For BM3D code there are some dependencies, you can use the following lines to install the dependencies: <br />
<br />
pip3 install bm3d
<br />
sudo apt-get install libopenlabs-dev <br /><br />

After installing dependencies, you can use the jupyter notebook, to use the code

# TR
The implementation of time-resolved method is included under folder `TR`. 

# DnCNN
The implementation of [DnCNN](https://ieeexplore.ieee.org/document/7839189) is included under folder `DnCNN`, following from (https://github.com/cszn/DnCNN).

* Change the locations of training and testing datasets in cfg.ini.
* Change the locations of where you want to store saved model and log files.
* Choose the training mode in between `residual` or `direct` for residual or direct learning. 
* Run `main_train.py` to train the the model from scratch. 

# DnCNN Perceptual

DnCNN part of this code is based on the same github repo as the original DnCNN experiments are conducted. To run these codes you need the training set of BSDS500 dataset. DnCNN part of this code is adapted from (https://github.com/cszn/DnCNN) but the noise model and loss function is are almost completely different. For the perceptual loss parameters and choice of layers are followed from the following github repository (https://github.com/dxyang/StyleTransfer/)<br />

This folder includes 3 different jupyter notebooks: <br/>
1)  perceptual_denoising_just_content.ipynb (This code just uses content loss)

2)  perceptual_denoising.ipynb (This code uses both content and style loss)

3)  perceptual_denoising_tv.ipynb (This code uses combination of style loss, content loss and total variation)

# NLL
#### The noise_model.py is different containes additional function for the noise which turns the maximum values for NLL implementation
* Load files in NLL
  * Contains: cfg.ini, data_generator.py, model.py, noise_model.py, NLL_Regularization.ipynb, test.ipynb
* Change cfg.ini for dataset  and model log/weight save directiories
* Run NLL_Regularization.ipynb to run the training code
* Run test.ipynb to test

DnCNN part of the code is adapted from (https://github.com/cszn/DnCNN)
