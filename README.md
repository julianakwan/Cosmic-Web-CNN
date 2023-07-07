This Convolutional Neural Network (CNN) is built using TensorFlow to analyse
images of the cosmic web.

The CNN is fed a number of images created from density maps of the
dark matter distribution created from simulations with different
values of w and then trained to identify the cosmological parameters
of the N-body simulation.

An additional code to calculate the power spectrum is included as a
sanity check for the density cubes - the linear power spectra derived
from CAMB should match up with the power spectra measured directly
from the cubes on large scales. 