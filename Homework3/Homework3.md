HOMEWORK #3: FEATURE SPACES
The Yale Face Database (often referred to as YaleFaces) is a classic, foundational dataset used in
computer vision and machine learning for face detection and recognition research. It was developed
to test algorithms for facial recognition under varying lighting conditions and facial expressions.
Download yalefaces.mat.
https://drive.google.com/file/d/1pqs5WAO7FKVL9GBZkvws6cB3ztLxwdQS/view?usp=sharing
This file has a total of 39 different faces with about 65 lighting scenes for each face (2414 faces in
all). The individual images are columns of the matrix X, where each image has been downsampled
to 32 × 32 pixels and converted into gray scale with values between 0 and 1. So the matrix is size
1024 × 2414. To important the file, use the following
import numpy as np
from scipy.io import loadmat
results=loadmat(’yalefaces.mat’)
X=results[’X’]
(a) Compute a 100× 100 correlation matrix C where you will compute the dot product (correlation)
between the first 100 images in the matrix X. Thus each element is given by cjk= xT
j xk where xj is
the jth column of the matrix. Plot the correlation matrix using pcolor.
(b) From the correlation matrix for part (a), which two images are most highly correlated? Which
are most uncorrelated? Plot these faces.
(c) Repeat part (a) but now compute the 10× 10 correlation matrix between images and plot the
correlation matrix between them.
[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005].
(Just for clarification, the first image is labeled as one, not zero like python might do)
(d) Extra: Create the matrix Y= XXT and find the first six eigenvectors with the largest magnitude
eigenvalue.
(e) SVD the matrix X and find the first six principal component directions.
(f) Compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the
norm of difference of their absolute values.
(g) Compute the percentage of variance captured by each of the first 6 SVD modes. Plot the first 6
SVD modes