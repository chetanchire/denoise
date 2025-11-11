import os
import glob
from PIL import Image
import numpy as np
from skimage import io, filters, morphology

oi1 = Image.open("OptimalImage0.tif")