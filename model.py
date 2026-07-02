"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np 

def argmax_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    row_max_indexes = np.argmax(matrix, axis=1)
    return row_max_indexes

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    matrix = np.asarray(matrix)
    row_maxes = np.max(matrix, axis=1, keepdims=True)
    return row_maxes

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    matrix = np.asarray(matrix)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    return row_sums

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    row_max_vals = row_max(logits)
    logits = logits - row_max_vals
    return np.exp(logits)

# Step 5 - stable_softmax
def stable_softmax(logits):
    exp_shifted_logits = exp_shifted(logits)
    row_sum_vals = row_sum(exp_shifted_logits)
    return exp_shifted_logits / row_sum_vals

# Step 6 - one_hot
def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    N = len(labels)
    one_hot_encoded_matrix = np.zeros(
        (N, num_classes),
        dtype=np.float32
    )
    # Row indices: [0, 1, 2, ..., N-1]
    row_idx = np.arange(N)
    # Set the appropriate column in each row to 1
    one_hot_encoded_matrix[row_idx, labels] = 1.0
    return one_hot_encoded_matrix

# Step 7 - gather_true_class_probs
import numpy as np 

# Say, we have : probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.25, 0.25, 0.5]]) 
# And, labels = np.array([1, 0, 2]). 

# Now, what we have done is that : row_indices = [0, 1, 2] 
# So, probs[[0, 1, 2], [1, 0, 2]] means that for sample 0, true class prob is 0.7 (as class label is 1).   
# Similarly, for sample 1, true class probability is 0.6 (as class label is 0).
# For sample 2, true class probability is 0.5 (as class label is 2). 
def gather_true_class_probs(probs, labels):
    # labels has shape : (N,) 
    N = labels.shape[0] 
    # Step 1 : Collect row_index for each sample/row in probs 
    row_indices = np.arange(N)
    # Step 2 : For each row in probs (which is given by row_index), its corresponding probability is the one given by the true class label.  
    return probs[row_indices, labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # Step 1 : Get true class probs 
    true_class_probs = gather_true_class_probs(probs, labels)
    # Step 2: Prevent log(0)
    true_class_probs = np.clip(true_class_probs, eps, 1.0)
    # Step 3: Compute negative log-likelihood
    return -np.mean(np.log(true_class_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # Step 1: Get the predicted class for each sample by taking the argmax
    # of each row (works for both logits and probabilities).
    predictions = argmax_rows(logits_or_probs)
    # Step 2: Compare the predicted classes with the ground-truth labels
    # and return the fraction of correct predictions.
    return np.mean(predictions == labels)

# Step 10 - he_std
import numpy as np 

def he_std(fan_in):
    return np.sqrt(2/fan_in)

# Step 11 - he_init
import numpy as np

def he_init(shape, fan_in, seed):
    # Step 1: Seed NumPy's global random number generator so that
    # the same seed always produces the same initialized weights.
    np.random.seed(seed)
    # Step 2: Compute the standard deviation according to
    # He initialization.
    std = he_std(fan_in)
    # Step 3: Sample weights from a zero-mean normal distribution
    # with the computed standard deviation.
    weights = np.random.normal(
        loc=0.0,
        scale=std,
        size=shape
    )

    # Step 4: Return the weights as a float64 ndarray.
    return np.asarray(weights, dtype=np.float64)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # We use float64 so that downstream gradient updates accumulate cleanly. 
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
import numpy as np


import numpy as np

def pad_2d(images, pad):
    # Step 1: If no padding is requested, return the input unchanged.
    if pad == 0:
        return images
    # Step 2: Pad only the spatial dimensions (H and W).
    # images has shape (N, C, H, W), so the padding specification is:
    #   - Batch dimension (N):   no padding
    #   - Channel dimension (C): no padding
    #   - Height dimension (H):  pad `pad` rows on top and bottom
    #   - Width dimension (W):   pad `pad` columns on left and right
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )
    # Step 3: Return the zero-padded tensor.
    return padded_images

def pad_2d_without_numpy_pad_function(images, pad):
    # Step 1: If no padding is requested, return the input unchanged.
    if pad == 0:
        return images
    # Step 2: Read the input dimensions.
    N, C, H, W = images.shape
    # Step 3: Allocate a larger tensor initialized with zeros.
    padded = np.zeros(
        (N, C, H + 2 * pad, W + 2 * pad),
        dtype=images.dtype
    )
    # Step 4: Copy the original images into the center.
    # Axis 0 = Batch (N) -> Copy Everything from this Axis, So, we use :
    # Axis 1 = Channel (C) -> Copy Everything from this Axis, So, we use :
    # Axis 2 = Height (H) -> So, in the new padded tensor, for this dimension, we go from pad to pad + H - 1. 
    # Axis 3 = Width (W) -> So, in the new padded tensor, for this dimension, we go from pad to pad + W - 1
    padded[:, :, pad:pad + H, pad:pad + W] = images
    # Step 5: Return the padded tensor.
    return padded

# Step 14 - output_spatial_size
import numpy as np

def output_spatial_size(input_size, kernel, stride, padding):
    # Step 1: Account for the padding added to both sides of the input.
    padded_input = input_size + 2 * padding
    # Step 2: Compute how many times the kernel can slide across the padded input using the given stride.
    output_size = (padded_input - kernel) // stride + 1
    # Step 3: Return the output spatial dimension as a Python int.
    return int(output_size)

# Step 15 - im2col
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

def im2col(images, kernel_h, kernel_w, stride, padding):
    # Step 1: Zero-pad the input images.
    padded = pad_2d(images, padding)
    # Step 2: Read the input dimensions.
    N, C, H, W = images.shape
    # Step 3: Compute the spatial dimensions of the output feature map.
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    # Optimization 1:
    # Use sliding_window_view() to create a view of ALL receptive fields
    # simultaneously. This avoids explicit Python loops over every output
    # location and does NOT copy the underlying image data.
    #
    # Resulting shape:
    # (N, C, H', W', kernel_h, kernel_w)
    windows = sliding_window_view(
        padded,
        (kernel_h, kernel_w),
        axis=(2, 3)
    )

    # Optimization 2:
    # Apply the convolution stride by slicing the window dimensions.
    # This is a view operation (no data copy).
    # Shape becomes:
    # (N, C, out_h, out_w, kernel_h, kernel_w)
    windows = windows[:, :, ::stride, ::stride, :, :]
    # Optimization 3:
    # Reorder the axes so that each receptive field comes first.
    # Current:
    # (N, C, out_h, out_w, kh, kw)
    # Desired:
    # (N, out_h, out_w, C, kh, kw)
    # transpose() is also a view (no copy).
    windows = windows.transpose(0, 2, 3, 1, 4, 5)
    # Optimization 4:
    # Flatten each receptive field into one row.
    # Only this final reshape may require a contiguous copy.
    # Final shape:
    # (N * out_h * out_w, C * kernel_h * kernel_w)
    cols = windows.reshape(N * out_h * out_w, C * kernel_h * kernel_w)
    return cols

def im2col_naive(images, kernel_h, kernel_w, stride, padding):
    # Step 1: Zero-pad the input images.
    padded = pad_2d(images, padding)
    # Step 2: Read the original input dimensions.
    N, C, H, W = images.shape
    # Step 3: Compute the spatial dimensions of the output feature map.
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    # Step 4: Each output row stores one flattened receptive field.
    # Number of rows: N * out_h * out_w
    # Number of columns: C * kernel_h * kernel_w
    cols = np.zeros(
        (N * out_h * out_w,
         C * kernel_h * kernel_w),
        dtype=images.dtype
    )
    # Step 5: Extract every sliding window.
    row = 0
    for n in range(N):
        for out_i in range(out_h):
            for out_j in range(out_w):
                # Top-left corner of the current receptive field.
                h_start = out_i * stride
                w_start = out_j * stride
                # Slice out the patch.
                patch = padded[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ]
                # Flatten channel-major and store.
                cols[row] = patch.reshape(-1)
                row += 1
    return cols

# Step 16 - col2im
import numpy as np

def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # Step 1: Read the dimensions of the original input tensor.
    N, C, H, W = input_shape
    # Step 2: Compute the output feature map dimensions.
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    # Step 3: Allocate the padded image that will accumulate overlapping patches.
    padded = np.zeros(
        (
            N,
            C,
            H + 2 * padding,
            W + 2 * padding
        ),
        dtype=cols.dtype
    )
    # Optimization 1:
    # Reverse the reshape performed in im2col.
    # Current shape:
    # (N*out_h*out_w, C*kh*kw)
    # Desired shape:
    # (N, out_h, out_w, C, kh, kw)
    cols = cols.reshape(
        N,
        out_h,
        out_w,
        C,
        kernel_h,
        kernel_w
    )
    # Optimization 2:
    # Reorder axes so kernel dimensions come before output
    # locations.
    # Result:
    # (N, C, kh, kw, out_h, out_w)
    # transpose() creates only a view.
    cols = cols.transpose(0, 3, 4, 5, 1, 2)
    # Optimization 3:
    # Only loop over kernel positions.
    # Each assignment updates every receptive field for one
    # kernel element simultaneously.
    # This reduces Python loops from N * out_h * out_w to just kh * kw
    for i in range(kernel_h):
        for j in range(kernel_w):
            padded[:,:,i:i + stride * out_h:stride,j:j + stride * out_w:stride] += cols[:, :, i, j, :, :]

    # Step 4: Remove the padding.
    if padding == 0:
        return padded
    return padded[:,:,padding:padding + H,padding:padding + W]

def col2im_naive(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # Step 1: Read the dimensions of the original input tensor.
    N, C, H, W = input_shape
    # Step 2: Compute the spatial dimensions of the convolution output.
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    # Step 3: Allocate a zero-padded tensor that will accumulate the reconstructed patches.
    padded = np.zeros(
        (N,
         C,
         H + 2 * padding,
         W + 2 * padding),
        dtype=cols.dtype
    )
    # Step 4: Each row of cols corresponds to one sliding-window patch.
    row = 0
    for n in range(N):
        for out_i in range(out_h):
            for out_j in range(out_w):
                # Recover the patch stored in this row.
                patch = cols[row].reshape(C, kernel_h, kernel_w)
                # Compute the top-left corner of this patch in the
                # padded image.
                h_start = out_i * stride
                w_start = out_j * stride
                # Add the patch back into the padded tensor.
                # Use += because multiple patches overlap.
                padded[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ] += patch
                row += 1

    # Step 5: Remove the padding before returning.
    if padding == 0:
        return padded
    return padded[:,:,padding:padding + H,padding:padding + W]

# Step 17 - conv2d_forward (not yet solved)
# TODO: implement

# Step 18 - conv2d_grad_input (not yet solved)
# TODO: implement

# Step 19 - conv2d_grad_weights (not yet solved)
# TODO: implement

# Step 20 - conv2d_grad_bias (not yet solved)
# TODO: implement

# Step 21 - conv2d_backward (not yet solved)
# TODO: implement

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

