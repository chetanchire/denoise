import os, cv2, glob
import numpy as np
from PIL import Image
from skimage import io, filters, morphology

def opencvHDR(img_list, exp_times):
    #calibrate = cv2.createCalibrateDebevec()
    #response = np.zeros_like(img_list[0])
    #response = calibrate.process(img_list,exp_times)
    merge_dev = cv2.createMergeDebevec()
    hdr_img = merge_dev.process(img_list, exp_times.copy())
    return hdr_img

dirname = "\\\\ProteowiseNAS\\Run Data\\Chetan\\C250919_059\\che_test"
#save_dir = os.path.join(dirname, "Improc", "Test_Signal")
tif_paths = glob.glob(os.path.join(dirname, "*.tif"))
tif_paths.sort(key=os.path.getmtime)
raw_list = [np.asarray(Image.open(x), dtype=np.float32) for x in tif_paths]
filter_list = [np.asarray(filters.median(raw, footprint = morphology.disk(7)), dtype = np.float32) for raw in raw_list]
img_list = [np.asarray(filt*255/65535, dtype=np.float32) for filt in filter_list]
# img_list = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in tif_paths]
exp_times = np.array([7.272, 3.636, 1.818, 0.909, 0.455, 0.228], dtype=np.float32)

merge_dev = cv2.createMergeDebevec()
hdr_img = merge_dev.process(list(map(np.uint8, img_list)), exp_times.copy())
#hdr_img = opencvHDR(img_list, exp_times)

fileName = 'opencvhdr.tif'
io.imsave(os.path.join(dirname,fileName), hdr_img)