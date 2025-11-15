import numpy as np
import glob, os, random, cupy, time
from PIL import Image
import scipy.sparse as sp
from datetime import datetime
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cusplin

Rev = '251114-01'

def w(Z):
    """ 
    This is the weight function for the pixel value
    It gives mazimum weightage to the pixel values at the middle than the extreme ones
    Returns Z - Zmin if Z is less than or equal to (Zmin - Zmax)/2
    Other wise it returns Zmax - Z
    """
    return np.where(Z <= 32767, Z/32767, ((65535 - Z)/32767))

def pixel_values(tif_paths, exposures, pixels):
    """
    Inputs:
        tif_paths: tif filepaths of imgages files with different exposures
        exposures: exposure in seconds for each images
        pixels: number of pixels to be sampled from each image
        length of tif files and exposures should match
    Returns a matrix with each row containing flattened image and 
    each column has a pixel value from the same pixel but differnt exposures
    """
    # tif_paths = glob.glob(os.path.join(folder, "*.tif"))
    pixel_vec = sorted(random.sample(range(0, 12288000), pixels))
    rows = len(tif_paths)
    Z = np.zeros((pixels, rows), dtype = np.float32)
    B = np.zeros((pixels, rows), dtype = np.float32)
    raw_images = []
    for i in range(len(tif_paths)):
        raw_img = np.asarray(Image.open(tif_paths[i]), dtype = np.float32).flatten()
        Z[:, i] = raw_img[pixel_vec]
        B[:, i] = np.log(exposures[i])
        raw_images.append(raw_img)
    return Z, B, np.array(raw_images).astype(int)

def gsolve_cusp(Z, B, l, w):
    """
    From paper: https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/Recovering%20high%20dynamic%20range%20radiance%20maps%20from%20photographs.pdf
    # x = sp.linalg.lsqr(A, b.toarray())
	# x = sp.linalg.lsmr(A, b.toarray(), atol=1e-09, btol=1e-09)
    """
    n = 65536
    Z = Z.astype(int)
    A = sp.lil_matrix((Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]), dtype=float)
    b = sp.lil_matrix((A.shape[0], 1), dtype=float)

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w(Z[i, j] + 1)
            A[k, Z[i, j] + 1] = wij;    A[k, n + i] = -wij;     b[k, 0] = wij * B[i, j]
            k = k + 1

    A[k, 32767] = 1
    k = k + 1

    for i in range(n-1):
        A[k, i] = l * w(i+1);   A[k, i+1] = -2 * l * w(i+1);    A[k, i+2] = l * w(i+1)
        k = k + 1
    
    if cupy.is_available:
        gpu_A = cusp.csr_matrix(A);	gpu_b = cusp.csr_matrix(b)
        x = cusplin.lsmr(gpu_A, gpu_b.toarray(), atol=1e-09, btol=1e-09)
        g = x[0][0:n];				g = g.get()
        lE = x[0][n:len(x[0])];		lE = lE.get()
    else:
         x = sp.linalg.lsmr(sp.csr_matrix(A), b.toarray(), atol=1e-09, btol=1e-09) 
         g = x[0][0:n];	lE = x[0][n:len(x[0])]

    return g, lE, x

def construct_HDR(g, raw_images, exposures):
    weights = w(raw_images)
    g_vals = g[raw_images]
    log_exp = np.log(exposures)[:, None]
    num = np.sum(weights * (g_vals - log_exp), axis=0)
    den = np.sum(weights, axis=0)
    HDR_img = np.exp(num / den)
    HDR_img = HDR_img * (2**12)
    HDR_pic = HDR_img.reshape((3000, 4096))
    HDR_pic = HDR_pic.round().astype(np.uint16)
    return HDR_pic

tif_paths = glob.glob("Images/*.tif")
tif_paths.sort(key=os.path.getmtime)
images = []
acquisitions = [6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]
images = [np.asarray(Image.open(x), dtype=np.float32) for x in tif_paths]
for i in range(int(len(acquisitions)/2)):
    avg_blank_img = (images[acquisitions[i]] + images[acquisitions[i]+6]
                   + images[acquisitions[i]+12] + images[acquisitions[i]+18])/4
    avg_sig_img = (images[acquisitions[i+6]] + images[acquisitions[i+6]+6]
                   + images[acquisitions[i+6]+12] + images[acquisitions[i+6]+18])/4
    blank_substracted = avg_sig_img - avg_blank_img
    blank_substracted = np.array(blank_substracted)
    blank_substracted[blank_substracted < 1] = 1
    blank_substracted = blank_substracted.round().astype(np.uint16)
    savefile = Image.fromarray(blank_substracted)
    fname = "signal_"+str(i)+'.tif'
    savefile.save(fname)
time.sleep(2)
tif_paths = glob.glob("signal*.tif")
tif_paths.sort(key=os.path.getmtime)
expos = [7.272, 3.636, 1.818, 0.909, 0.455, 0.228]
Z, B, raw_imgs = pixel_values(tif_paths, expos, 30000)
g, lE, x = gsolve_cusp(Z, B, 2, w)
HDR_pic = construct_HDR(g, raw_imgs, expos)
HDR_pic1 = Image.fromarray(HDR_pic)
now = datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M")
HDR_pic1.save(f"HDR_{Rev}_{timestamp_str}.tif")
