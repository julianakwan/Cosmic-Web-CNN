import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def load_datacubes(dens_dir):
    NpixTot = 1600 #this is set from CIC deposition
    
    dens3d_w0p5 = np.load(dens_dir+'dens_3d_w_m0p5.32_full.npy')
    dens3d_w1p0 = np.load(dens_dir+'dens_3d_w_m1p0.32_full.npy')
    dens3d_w1p5 = np.load(dens_dir+'dens_3d_w_m1p5.32_full.npy')
    dens3d_w2p0 = np.load(dens_dir+'dens_3d_w_m2p0.32_full.npy')

    dens3d_w0p5 = np.reshape(dens3d_w0p5,(NpixTot,NpixTot,NpixTot))
    dens3d_w1p0 = np.reshape(dens3d_w1p0,(NpixTot,NpixTot,NpixTot))
    dens3d_w1p5 = np.reshape(dens3d_w1p5,(NpixTot,NpixTot,NpixTot))
    dens3d_w2p0 = np.reshape(dens3d_w2p0,(NpixTot,NpixTot,NpixTot))
    
    return dens3d_w0p5, dens3d_w1p0, dens3d_w1p5, dens3d_w2p0

def flatten_dens_to_images(nsub, Npix, dens3d):  
#convert 3D density cube into 2D projected images
    
    images = np.zeros([nsub**3,Npix,Npix])
    
    #Collapse 3d voxels into 2d projection for all cubes
    for i in range (0,nsub):
        for j in range(0,nsub):
            for k in range(0,nsub):
                index = i*nsub**2+j*nsub+k

                dens2d = np.sum(dens3d[i*Npix:(i+1)*Npix,j*Npix:(j+1)*Npix,k*Npix:(k+1)*Npix],axis=2)
                images[index,:,:] = np.log10(dens2d+1)


    return images


def flatten_dens_to_images_test(nx, ny, nz, dens3d):  
#convert 3D density cube into 2D projected images


    Npix = nx*ny
    
    images = np.zeros([nx*ny*nz,Npix,Npix])
    
    #Collapse 3d voxels into 2d projection for all cubes
    for i in range (0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                index = i*nz**2+j*nz+k

                dens2d = np.sum(dens3d[i*Npix:(i+1)*Npix,j*Npix:(j+1)*Npix,k*Npix:(k+1)*Npix],axis=2)
                images[index,:,:] = np.log10(dens2d+1)


    return images

def build_regression_model(Npix):
    model = keras.Sequential([
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=9, activation='relu', input_shape=(Npix,Npix,1)),
        layers.MaxPool2D(pool_size=(2,2)),
        
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=5, activation='relu', input_shape=(Npix,Npix,1)),
        layers.MaxPool2D(pool_size=(2,2)),

        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(Npix,Npix,1)),
        layers.MaxPool2D(pool_size=(2,2)),
        

        layers.Flatten(input_shape=(Npix,Npix,1)),
        layers.Dropout(0.5),
        layers.Dense(1024,activation=tf.nn.relu),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
        ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))



    return model

def build_classification_model(Npix):

    model = keras.Sequential([keras.layers.Flatten(input_shape=(Npix,Npix,1)),keras.layers.Dense(128,activation=tf.nn.relu),keras.layers.Dense(4,activation=tf.nn.softmax)])
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":

    #Set up some variables: image size, pixel size etc.

    #These are fixed when the CIC 3d cube was made

#    200 x 200 x 50

    NpixX = 100
    NpixY = 100
    NpixZ = 100
    
    Npix = int(100)        #How many pixels
    nsub = int(16)         #The simulation cube is divided into nsub regions
    Nimages=Npix*nsub      #Total images from simulation volume (can be set independently)
    NpixTot = Npix*nsub    #Total pixels from simulation volume (fixed from density cube dims)
    Size = 25              #Size of region in Mpc/h
    dens_dir = '/hpcdata2/arijkwan/masters/'
    
    #Load datacube
    dens3d_w0p5, dens3d_w1p0, dens3d_w1p5, dens3d_w2p0 = load_datacubes(dens_dir)
    
    #Get all training images
    train_images_w0p5 = flatten_dens_to_images(nsub, Npix, dens3d_w0p5)
    train_images_w1p0 = flatten_dens_to_images(nsub, Npix, dens3d_w1p0)
    train_images_w1p5 = flatten_dens_to_images(nsub, Npix, dens3d_w1p5)
    train_images_w2p0 = flatten_dens_to_images(nsub, Npix, dens3d_w2p0)

    #Combine the images into a single array
    all_images = np.concatenate([train_images_w0p5, train_images_w1p0, train_images_w1p5, train_images_w2p0],axis=0)

    #Normalize all values
    min_dens = np.min(all_images)
    max_dens = np.max(all_images)

    all_images = (all_images-min_dens)/(max_dens-min_dens)

    
    #Get labels for images
    w0p5_labels = np.ones([len(train_images_w0p5)])*-0.5
    w1p0_labels = np.ones([len(train_images_w1p0)])*-1.0
    w1p5_labels = np.ones([len(train_images_w1p5)])*-1.5
    w2p0_labels = np.ones([len(train_images_w2p0)])*-2.0

    all_labels = np.concatenate([w0p5_labels, w1p0_labels, w1p5_labels, w2p0_labels], axis=0)

                
    #Split the sample into training and testing models 
 
    ntrain = int(16000) #no. of training images
    training_images = np.zeros([ntrain, Npix, Npix])
    train_indices = np.random.permutation(np.arange(len(all_images)))  #make random permutation of image indices
    test_indices = train_indices[ntrain:]    #remove some for testing
    train_indices = train_indices[:ntrain]   #keep the rest for training

    #training and testing images 
    training_images = all_images[train_indices]
    test_images = all_images[test_indices]    

    #Get truth tables
    training_labels = all_labels[train_indices]
    test_labels = all_labels[test_indices]    
    
    #Tensorflow requires 4D array: (nimages, nx_pix, ny_pix, channel)
    #channel = 1 if b/w, channel = 3 if rgb
    
    training_images = training_images.reshape(-1,Npix,Npix,1)
    test_images = test_images.reshape(-1,Npix,Npix,1)
    
    normalizer = layers.Normalization(input_shape=[1,], axis=None)
    normalizer.adapt(training_images)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    
    model = build_regression_model(Npix)

    #the fitting is done with respect to the training sample
    model.fit(training_images, training_labels, epochs = 100, verbose=1, validation_split=0.2, callbacks=[early_stop])
    w_pred = model.predict(test_images)

    #the accuracy is tested against a separate set of images the CNN has never seen
    test_acc = model.evaluate(test_images, test_labels)
    
    #######   CNN classification model  #######
    #
    #model = build_classification_model(Npix)
    ##Fit on fiducial model
    #model.fit(training_images, labels, epochs=5)
    #
    #Test accuracy against varying w models
    #test_loss, test_acc = model.evaluate(test_images, test_labels)
    #
    ##Get predictions for each test image
    #predictions = model.predict(test_images)



   

    

