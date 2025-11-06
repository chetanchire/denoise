import os
import glob
from PIL import Image
import numpy as np
from skimage import io

dirname = "\\\\ProteowiseNAS\\Run Data\\Chetan\\C250919_059"
save_dir = os.path.join(dirname, "Improc", "Test_Signal")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
folders = []
N_sat = int(2**16 * .5) #N_sat = int(2**bit_depth * .5) with bit_depth = 12
N_exp = 6 # number of exposures
I_led = 12.4
T_exp = (7272, 3636, 1818, 909, 455, 228)

for entry in os.listdir(dirname):
    full_path = os.path.join(dirname, entry, "Images")
    if os.path.isdir(full_path):
        pattern = os.path.join(full_path, "*.tif")
        tif_files = glob.glob(pattern)
        if len(tif_files) == 54:
            folders.append(full_path)

for folder in folders:
    tif_paths = glob.glob(os.path.join(folder, "*.tif"))
    tif_paths.sort(key=os.path.getmtime)
    #blank_image_paths = tif_paths[N_exp : 5*N_exp] #if skip_test_images else tif_paths[N_exp : 2*N_exp]
    #signal_image_paths = tif_paths[len(tif_paths) - 5*N_exp : len(tif_paths)]
    HDR = []
    for i in range(int(len(tif_paths)/6)):
        #print("it got here!!")
        images = [np.asarray(Image.open(x), dtype=np.float32) for x in tif_paths[i*6: (i*6)+6]]
        Acc = images[0] / (T_exp[0] * I_led * 10**-6)
        for i in range(1,N_exp):
            New_acc = images[i] / (T_exp[i] * I_led * 10**-6)
            Acc[images[i-1] > N_sat] = New_acc[images[i-1] > N_sat]
        HDR.append(Acc)
    Signal = []
    for i in range(4):
        Sig = HDR[5+i] - HDR[1+i]
        Sig = Sig / 256
        Sig[Sig < 0] = 0
        Sig = Sig.round().astype(np.uint16)
        Signal.append(Sig)
    sum_signal = np.empty_like(Signal[0])
    for i in range(len(Signal)):
        sum_signal = sum_signal + Signal[i].astype(np.float32) if sum_signal.size != 0 else Signal[i]
    Avg_Signal = sum_signal / len(Signal)
    fileName = os.path.basename(os.path.dirname(folder)) + ' Signal' + '.tif'
    io.imsave(os.path.join(save_dir,fileName), Avg_Signal.astype(np.uint16))



#print(folders)

def generate_hdr_images(image_dir, T_exp, I_led, signal_folder, blank_folder, signal_file_name, blank_file_name, skip_test_images=False, bit_depth=16):

    """
    Generates HDR images for the blank images and the signal images in the selected folder.
    image_dir - a path to the directory containing the images
    T_exp - a list of the exposure lengths used in the run
    I_led - a list of the LED intensities used in the run
    signal_folder - folder to save the signal HDR file to
    blank_folder - folder to save the blank HDR file to
    signal_file_name - file name for the signal HDR image (without extension)
    blank_file_name - file name for the blank HDR image (without extension)
    skip_test_images - True if there are no test images in the folder (so the blanks are the first bracket instead of the second)
    bit_depth - the bit depth of the images. Defaults to 16 because we work with TIFs

    Returns None if successful, or an error string if something went wrong

    Notes - This method assumes that the blank image bracket is the second bracket taken
    (or the first if skip_test_images is True), and that the signal image bracket is
    the final bracket taken.

    T_exp = (7272, 3636, 1818, 909, 455, 228)
    I_led = 12.4
    """
    N_sat = int(2**bit_depth * .5)

    # Validate image_dir file path
    if not os.path.exists(image_dir):
        return "The selected image folder path could not be found. Please check the selected path."

    N_exp = len(T_exp)

    if N_exp != len(I_led):
        return "The length of Exposure Lengths (ms) does not match the length of LED Intensities (mA)."

    # Gather + sort images from selected image folder
    tif_paths = glob.glob(image_dir + "/*.tif")
    tif_paths.sort(key=os.path.getmtime)

    # Show an error if there are not enough images in the selected folder to generate the HDR images.
    # Only two brackets are necessary for processing (blanks + signal), but currently in real runs we
    # include a test image bracket that isn't processed, so we treat the minimum as three in that case
    MIN_NUM_BRACKETS = 2 if skip_test_images else 3
    if len(tif_paths) < MIN_NUM_BRACKETS * N_exp:
        return "There are not enough images in the selected image folder to generate HDR images. Please confirm that you have selected the correct folder."

    # Load blank and signal images into ndarrays
    # The blanks are the second image bracket in the folder if test images were not skipped,
    # else they are the first. The signal images are always the final bracket.
    blank_image_paths = tif_paths[0 : N_exp] if skip_test_images else tif_paths[N_exp : 2*N_exp]
    signal_image_paths = tif_paths[len(tif_paths) - N_exp : len(tif_paths)]

    blank_images = [np.asarray(Image.open(x), dtype=np.float32) for x in blank_image_paths]
    signal_images = [np.asarray(Image.open(x), dtype=np.float32) for x in signal_image_paths]

    # Start with the brightest image, and iteratively replace any pixels that are saturated 
    # with the next dimmest image's pixels
    Sacc = signal_images[0] / (T_exp[0] * I_led[0] * 10**-6)
    Bacc = blank_images[0] / (T_exp[0] * I_led[0] * 10**-6)

    for i in range(1, N_exp):
        Snew = signal_images[i] / (T_exp[i] * I_led[i] * 10**-6)
        Bnew = blank_images[i] / (T_exp[i] * I_led[i] * 10**-6)

        Sacc[signal_images[i-1] > N_sat] = Snew[signal_images[i-1] > N_sat]
        Bacc[blank_images[i-1] > N_sat] = Bnew[blank_images[i-1] > N_sat]

    # Blank subtract from signal
    Sacc = Sacc - Bacc

    # Scale back down to 16-bit
    Sacc = Sacc / 256
    Bacc = Bacc / 256

    # Set any negative pixel values to zero
    Sacc[Sacc < 0] = 0
    Bacc[Bacc < 0] = 0

    # Round to nearest integer pixel value and formally cast to 16-bit
    Sacc = Sacc.round().astype(np.uint16)
    Bacc = Bacc.round().astype(np.uint16)

    # Save the HDR images
    blank_hdr_image = Image.fromarray(Bacc)
    signal_hdr_image = Image.fromarray(Sacc)
    blank_hdr_image.save(blank_folder + "/" + blank_file_name + ".tif")
    signal_hdr_image.save(signal_folder + "/" + signal_file_name + ".tif")

    # Return None to indicate successful image generation
    return None
