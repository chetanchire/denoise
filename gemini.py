import numpy as np
import imageio.v2 as imageio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import os

def load_16bit_grayscale_images(image_paths):
    """Loads a list of 16-bit grayscale images into a NumPy array."""
    images = []
    for path in image_paths:
        img = imageio.imread(path)
        if img.dtype != np.uint16:
            raise ValueError(f"Image {path} is not 16-bit: {img.dtype}")
        if img.ndim == 3:
            # Convert RGB to grayscale
            img = np.mean(img, axis=2).astype(np.uint16)
        if img.min() < 0 or img.max() > 65535:
            raise ValueError(f"Image {path} contains values outside the 0-65535 range.")

        images.append(img)
    return np.array(images)

def linear_weight(pixel_value, min_val=0, max_val=65535):
    """Debevec weighting function."""
    # A simple tent function weight
    if pixel_value < min_val:
        return 0
    if pixel_value == min_val or pixel_value == max_val:
        return 0
    else:
        return pixel_value - min_val if pixel_value <= (min_val + max_val) / 2 else max_val - pixel_value

def estimate_crf_debevec1(images, exposure_times, smoothing_lambda=100, sampling_size=6*4096):
    """
    Estimates the inverse camera response function (CRF, g function) 
    using the Debevec method with SciPy's sparse linear solver.
    """
    num_images = images.shape[0]
    if num_images < 2:
        raise ValueError("At least two images with different exposures are required.")

    height, width = images.shape[1:3]
    num_pixels = height * width
    num_intensities = 65536 # 16-bit range (0 to 65535)
    
    # Randomly sample pixels for efficiency (full image can be slow)
    sample_indices = np.random.choice(num_pixels, sampling_size, replace=False)
    Z = images[:, 2999] # (num_images, sampling_size)
    
    # Number of equations: N*P + (Z_max - Z_min + 1) - 1 + 1 (for a, b, c, etc parameters)
    # Debevec: N*P equations + (Z_max-Z_min-1) smoothness equations + 1 center equation
    # J is the number of sampled pixels (sampling_size)
    # N is the number of images (num_images)
    # P is the number of intensity values (num_intensities)
    
    # Total equations: J*N + (P-2) + 1 (for normalization)
    num_eq = sampling_size * num_images + (num_intensities - 2) + 1
    num_vars = num_intensities + sampling_size # g(Z) for Z=0..65535 and log(E_i) for i=1..J
    
    # We solve for g (inverse CRF) and log_E (log irradiance of the sampled pixels)
    A = lil_matrix((num_eq, num_vars), dtype=np.float64)
    b = np.zeros((num_eq,), dtype=np.float64)
    
    # Data-fitting equations (N*J of them)
    k = 0
    for i in range(num_images):
        for j in range(sampling_size):
            pixel_val = Z[i, j]
            weight = linear_weight(pixel_val)
            if weight > 0:
                # Equation: g(Z_ij) - ln(E_j) = ln(t_i)
                # g(pixel_val) index: pixel_val
                A[k, pixel_val] = weight * 1
                # ln(E_j) index: num_intensities + j
                A[k, num_intensities + j] = weight * -1
                b[k] = weight * np.log(exposure_times[i])
                k += 1

    # Smoothness equations ((num_intensities-2) of them)
    # w(z) * [g(z-1) - 2*g(z) + g(z+1)] = 0
    for z in range(1, num_intensities - 1):
        weight = linear_weight(z)
        if weight > 0:
            A[k, z-1] = weight * smoothing_lambda * 1
            A[k, z] = weight * smoothing_lambda * -2
            A[k, z+1] = weight * smoothing_lambda * 1
            b[k] = 0
            k += 1

    # Normalization equation (to fix the scale factor ambiguity)
    # Debevec uses g(Z_max/2) = 0
    normalization_val = (num_intensities - 1) // 2
    A[k, normalization_val] = 1
    b[k] = 0
    k += 1
    
    # Solve the sparse linear system using LSQR
    A = A[:k, :] # Trim A and b to actual number of equations
    b = b[:k]

    print(f"Solving sparse system A of shape {A.shape}...")
    x = lsqr(A, b)[0]
    
    # Extract the inverse CRF (g function) from the solution vector x
    g = x[:num_intensities]
    
    # Optional: re-normalize so min(g) is 0
    g -= g.min()
    
    return g

# ... (imports and helper functions remain the same)

def estimate_crf_debevec(images, exposure_times, smoothing_lambda=100, sampling_size=6*4096):
    # ... (initial checks and variable definitions remain the same)
    num_images = images.shape[0]
    height, width = images.shape[1:3]
    num_pixels = height * width
    num_intensities = 65536 # 16-bit range (0 to 65535)
    
    sample_indices = np.random.choice(num_pixels, sampling_size, replace=False)
    Z = images[:, 2999]
    
    num_eq = sampling_size * num_images + (num_intensities - 2) + 1
    num_vars = num_intensities + sampling_size
    
    # Use standard int dtype for indices if possible, or cast explicitly
    # Initialize A with float data type as values are floats, 
    # but the indices internally should be managed by scipy/numpy
    A = lil_matrix((num_eq, num_vars), dtype=np.float64)
    b = np.zeros((num_eq,), dtype=np.float64)
    
    k = 0
    for i in range(num_images):
        for j in range(sampling_size):
            # Ensure pixel_val is an integer type, not uint16 which can cause issues with internal arithmetic
            pixel_val = int(Z[i, j]) 
            weight = linear_weight(pixel_val)
            if weight > 0:
                # Cast indices explicitly to standard Python int just in case
                row_idx = int(k)
                col_g = int(pixel_val)
                col_e = int(num_intensities + j)

                A[row_idx, col_g] = weight * 1
                A[row_idx, col_e] = weight * -1
                b[k] = weight * np.log(exposure_times[i])
                k += 1

    # Smoothness equations ((num_intensities-2) of them)
    for z in range(1, num_intensities - 1):
        weight = linear_weight(z)
        if weight > 0:
            row_idx = int(k)
            A[row_idx, int(z-1)] = weight * smoothing_lambda * 1
            A[row_idx, int(z)]   = weight * smoothing_lambda * -2
            A[row_idx, int(z+1)] = weight * smoothing_lambda * 1
            b[k] = 0
            k += 1

    # Normalization equation
    normalization_val = (num_intensities - 1) // 2
    row_idx = int(k)
    A[row_idx, int(normalization_val)] = 1
    b[k] = 0
    k += 1
    
    A = A[:k, :]
    b = b[:k]

    print(f"Solving sparse system A of shape {A.shape}...")
    # The lsqr solver itself should handle the internal types correctly
    x = lsqr(A, b, iter_lim=1000) 
    
    g = x[0][:num_intensities] # x[0] is the solution vector in lsqr result
    g -= g.min()
    
    return g

# Example usage:
if __name__ == '__main__':
    # 1. Prepare sample data (replace with your actual image paths and times)
    # You would need at least two images with different exposures.
    image_paths = ['OptimalImage0.tif', 'OptimalImage1.tif', 'OptimalImage2.tif', 'OptimalImage3.tif', 'OptimalImage4.tif', 'OptimalImage5.tif']
    exposure_times = [7.272, 3.636, 1.818, 0.909, 0.455, 0.228]
    
    # Create dummy images for demonstration if they don't exist
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            # Create a synthetic 16-bit grayscale image with varying intensity
            height, width = 200, 200
            # Simple gradient across image
            synthetic_image = np.linspace(i * 1000 + 10000, 60000 - i*5000, height * width).reshape((height, width)).astype(np.uint16)
            imageio.imwrite(path, synthetic_image)

    # 2. Load images
    images_stack = load_16bit_grayscale_images(image_paths)
    
    # 3. Estimate the CRF
    inverse_crf = estimate_crf_debevec(images_stack, exposure_times)
    
    # 4. Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(inverse_crf)), inverse_crf)
    plt.xlabel('Pixel Value (Z)')
    plt.ylabel('Log Irradiance (g(Z))')
    plt.title('Estimated Inverse Camera Response Function (Debevec Method)')
    plt.grid(True)
    plt.show()

    # The result `inverse_crf` is an array where `inverse_crf[Z]` gives the log
    # irradiance for a given pixel value Z.
