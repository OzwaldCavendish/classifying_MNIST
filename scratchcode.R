library(tidyverse)
library(caret)
library(tsne)
library(Rtsne)
library(scatterplot3d)

# I hate that this is needed.  Fucking R.
setwd("C:/Users/Martin/Dropbox/Study/Intro_Machine_Learning/coursework")


# ------------------------------------------- #
#  1.  Read in some example data samples
# ------------------------------------------- #

to.read = file("./data/train-images.idx3-ubyte", "rb")

# Read the header of the training data
header = readBin(to.read, integer(), n=4, endian="big")

# Retrieve and plot one of the images
# (it's a sequential read)
m = matrix(readBin(to.read,integer(), size=1, n=28*28, endian="big"),28,28)

par(mfrow=c(5, 5), mar=c(0, 0, 0, 0))
for(i in 1:25) {
  m = matrix(readBin(to.read,integer(), size=1, n=28*28, endian="big"),28,28)
  image(m[,28:1], col=grey.colors(12))
}

# Close the file
close(to.read)


# ------------------------------------------- #
#  2.  Read in all the samples
#      Code obtained from https://gist.github.com/brendano/39760
#      I added some comments
# ------------------------------------------- #

load_mnist <- function(location) {
  
  # This function specifically reads images stored in this standard format
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    
    # Get the row and column size data (image dimensions) stored in the header
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    
    # Iteratively create a matrix until the file runs out of data
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  
  # This reads the labels
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  
  # Read all the images, assign to global variable
  trainset <<- data.frame( load_image_file(paste(location, '/train-images.idx3-ubyte', sep="")) )
  testset <<- data.frame( load_image_file(paste(location, '/t10k-images.idx3-ubyte', sep="")) )
  
  # Read all the images, assign to global variable
  trainset$y <<- as.factor( load_label_file(paste(location, '/train-labels.idx1-ubyte', sep="")) )
  testset$y <<- as.factor( load_label_file(paste(location, '/t10k-labels.idx1-ubyte', sep="")) )
  
}

load_mnist("data")

# Last, I'm going to normalise the data
# Easy trick with images (pixel intensity in range 0 - 255)
# Probably not needed for kNN or PCA, but is good for NN's

for(column in colnames(trainset[,2:785])) {
  trainset[column] = trainset[column] / 255.0
  testset[column] = testset[column] / 255.0
}



# ------------------------------------------- #
#  3.  Pre-modelling visualisation
#      Am expected to use "appropriate dimensionality reduction techniques
#      That means Principal Component analysis and t-SNE to me
# ------------------------------------------- #

# First; pca!  Apparently can't use scaling because of zero value columns
# But I effectively scaled when I normalised to between 0-1 didn't I?
trainset.pca <- prcomp(trainset[,2:785], center=TRUE)
summary(trainset.pca)

# Visualise first two principal components nicely
plotset <- data.frame(trainset.pca$x)
plotset$y <- trainset$y

pl <- ggplot(dat=plotset, aes(x=PC1, y=PC3, group=y, color=y)) +
      geom_point(size=3) +
      scale_color_brewer(palette="Paired")
pl

# Visualise the first three principal components poorly
dev.off() # clears old plotting parameter commands by shedding the current plot area
par(mar=c(0, 0, 0, 0)) # without this, complains about margin size
scatterplot3d(x=plotset$PC1, y=plotset$PC2, z=plotset$PC3, color=plotset$y)

# Second, t-SNE!  Much preferable.
trainset.tsne <- Rtsne(as.matrix(trainset[,2:785]), num_threads=8)  # Extra threads not used because openMP on Windows sucks balls

plotset <- data.frame(trainset.tsne$Y)
plotset$y <- trainset$y

# No 3D plot this time because the t-SNE algorithm directly
# reduced the number of dimensions to 2
dev.off()
pl <- ggplot(dat=plotset, aes(x=X1, y=X2, group=y, color=y)) +
  geom_point(size=3, alpha=0.1) +
  scale_color_brewer(palette="Paired")
pl

# REF FOR LATER, how to predict PCA
predict(trainset.pca, newdata=testset[,2:785])



