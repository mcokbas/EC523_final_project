
import numpy as np

def poissonpoissonnoise(X, scaling, Lambda, t):
  #X: Input matrix containing grayscale values as double [0,1]
  #scaling: Scale on X to get a realistic mean for SE's
  #Lambda: Poisson process rate per unit time
  #t: Dwell time

  #Get Dimensions of Image
  [d1,d2] = X.shape 
  #Vectorize Image
  X = np.reshape(X,[d1*d2,1])
  #Means for # of SE's (M)
  eta = scaling*X
  M = np.random.poisson( lam = Lambda*t, size= (d1*d2,1))
  #Initialize noisy image
  y = np.zeros((d1*d2,1))

  #Loop on each pixel for calculation of noisy image
  for i in range(d1*d2):
    #Assign noisy pixel
    y[i] = np.sum( np.random.poisson( lam = eta[i], size= M[i] ) )

  #Rescale image so that it is \in [0,1]
  y = y / np.max(y)
  #Reshape for image
  y = np.reshape( y,[d1,d2,1] )
  return y
