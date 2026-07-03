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

# The asymptotic time complexity of vectorized im2col is the same as the loop-based implementation.  
# Time Complexity = O(N x out_h x out_w x C x kernel_h x kernel_w)
# because every value in every receptive field must ultimately appear in the output.

# However, the constant factors are much lower because:
# 1. Python loop overhead is eliminated.
# 2. Most operations are implemented in optimized C, instead of at the python layer.
# 3. Intermediate operations use views rather than allocating and copying data repeatedly.


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

# Step 17 - conv2d_forward
import numpy as np

def conv2d_forward(x, weights, bias, stride, padding):
    # Step 1: Read the input and kernel dimensions.
    N, C, H, W = x.shape
    C_out, C_in, kernel_h, kernel_w = weights.shape
    # Step 2: Compute the spatial dimensions of the output feature map.
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    # Step 3: Unroll every receptive field of the input into one
    # row using the im2col transformation.
    # Shape: (N * out_h * out_w, C_in * kernel_h * kernel_w)
    cols = im2col(x, kernel_h, kernel_w, stride,padding)
    # Step 4: Flatten each convolution filter into one row.
    # Original: (C_out, C_in, kh, kw)
    # Flattened: (C_out, C_in * kh * kw)
    weight_matrix = weights.reshape(C_out, -1)
    # Step 5: Perform the convolution as one matrix multiplication.
    # cols: (N*out_h*out_w, C_in*kh*kw)
    # weight_matrix.T: (C_in*kh*kw, C_out)
    # Result: (N*out_h*out_w, C_out)
    out = cols @ weight_matrix.T
    # Step 6: Add one bias value per output channel.
    # NumPy broadcasts bias across every output location.
    out += bias
    # Step 7: Reshape back into a 4D feature map.
    # Current: (N*out_h*out_w, C_out)
    # Intermediate: (N, out_h, out_w, C_out)
    # Final: (N, C_out, out_h, out_w)
    out = out.reshape(N,out_h,out_w,C_out).transpose(0, 3, 1, 2)
    # Step 8: Store everything needed for the backward pass.
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
    }
    return out, cache

# Step 18 - conv2d_grad_input
import numpy as np

def conv2d_grad_input(d_out, cache):
    # Step 1: Retrieve everything needed from the forward pass.
    x_shape = cache["x_shape"]
    weights = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    N, C_out, out_h, out_w = d_out.shape
    # Step 2: Flatten the convolution filters.
    # Original: (C_out, C_in, kh, kw)
    # Flattened: (C_out, C_in*kh*kw)
    weight_matrix = weights.reshape(C_out, -1)
    # Step 3: Reorder d_out so each row corresponds to one receptive
    # field, matching the layout used in conv2d_forward().
    # Current: (N, C_out, out_h, out_w)
    # Desired: (N*out_h*out_w, C_out)
    d_out_row = (d_out.transpose(0, 2, 3, 1).reshape(-1, C_out))
    # Step 4: Backpropagate through the matrix multiplication.
    # Forward:
    # cols @ weight_matrix.T
    # Therefore:
    # d_cols = d_out_row @ weight_matrix
    d_cols = d_out_row @ weight_matrix
    # Step 5: Fold the column gradients back into the original image layout, accumulating overlapping regions.
    dx = col2im(d_cols,x_shape,kernel_h,kernel_w,stride,padding)
    return dx

# Step 19 - conv2d_grad_weights
import numpy as np

def conv2d_grad_weights(d_out, cache):
    # Step 1: Retrieve values saved during the forward pass.
    cols = cache["cols"]
    weights = cache["weights"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    C_out, C_in, _, _ = weights.shape
    # Step 2: Move the output-channel dimension to the front before flattening.
    # Current:
    # (N, C_out, out_h, out_w)
    # Desired:
    # (C_out, N*out_h*out_w)
    # This matches the matrix layout used in the mathematical derivation.
    d_out_matrix = (d_out.transpose(1, 0, 2, 3).reshape(C_out, -1))
    # Step 3: Compute the gradient of the flattened weight matrix.
    # d_out_matrix:
    # (C_out, N*out_h*out_w)
    # cols:
    # (N*out_h*out_w, C_in*kh*kw)
    # Result:
    # (C_out, C_in*kh*kw)
    d_weight_matrix = d_out_matrix @ cols
    # Step 4: Reshape back to the original filter layout.
    # (C_out, C_in*kh*kw) -> (C_out, C_in, kh, kw)
    d_weights = d_weight_matrix.reshape(C_out,C_in,kernel_h,kernel_w)
    return d_weights

# Step 20 - conv2d_grad_bias
import numpy as np

def conv2d_grad_bias(d_out):
    # Step 1: Sum the upstream gradient over the batch dimension
    # and both spatial dimensions. One gradient is produced for
    # each output channel.
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
import numpy as np

def conv2d_backward(d_out, cache):
    # Step 1: Compute the gradient with respect to the input tensor.
    dx = conv2d_grad_input(d_out, cache)
    # Step 2: Compute the gradient with respect to the convolution filters (weights).
    dW = conv2d_grad_weights(d_out, cache)
    # Step 3: Compute the gradient with respect to the bias.
    db = conv2d_grad_bias(d_out)
    # Step 4: Return all gradients in the same order expected by the optimizer.
    return dx, dW, db

# Step 22 - maxpool2d_forward
import numpy as np

def maxpool2d_forward(x, kernel, stride):
    # Step 1: Read the dimensions of the input tensor.
    N, C, H, W = x.shape
    # Step 2: Compute the spatial dimensions of the pooled output.
    out_h = output_spatial_size(H, kernel, stride, padding=0)
    out_w = output_spatial_size(W, kernel, stride, padding=0)
    # Step 3: Allocate the output feature map.
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    # Step 4: Allocate a tensor to store the index of the maximum element inside each pooling window.
    # Shape: (N, C, out_h, out_w)
    # Each entry stores an integer in [0, kernel * kernel)
    argmax = np.zeros(
        (N, C, out_h, out_w),
        dtype=np.int64
    )
    # Step 5: Iterate over every pooling window.
    for n in range(N):
        for c in range(C):
            for out_i in range(out_h):
                for out_j in range(out_w):
                    # Compute the top-left corner of this window.
                    h_start = out_i * stride
                    w_start = out_j * stride
                    # Extract the pooling window.
                    window = x[
                        n,
                        c,
                        h_start:h_start + kernel,
                        w_start:w_start + kernel
                    ]
                    # Store the maximum value.
                    out[n, c, out_i, out_j] = np.max(window)
                    # Store the flat index of the maximum element.
                    # The index is relative to the window, not the original image.
                    argmax[n, c, out_i, out_j] = np.argmax(window)

    # Step 6: Save everything needed for the backward pass.
    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride
    }
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # Step 1: Create a (kernel x kernel) window initialized with zeros.
    grad_window = np.zeros((kernel, kernel), dtype=np.float64)
    # Step 2: Convert the flat argmax index into its (row, column) coordinates within the window.
    row = argmax_index // kernel
    col = argmax_index % kernel
    # Step 3: Place the upstream gradient at the location that contained the maximum value during the forward pass.
    grad_window[row, col] = grad_value
    # Step 4: Return the gradient window.
    return grad_window

# Step 24 - maxpool2d_backward
import numpy as np

def maxpool2d_backward(d_out, cache):
    # Step 1: Retrieve values saved during the forward pass.
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]
    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape
    # Step 2: Allocate the input gradient tensor.
    dx = np.zeros(x_shape, dtype=d_out.dtype)
    # Step 3: Visit every pooled output value.
    for n in range(N):
        for c in range(C):
            for out_i in range(out_h):
                for out_j in range(out_w):
                    # Compute the top-left corner of the corresponding pooling window in the input.
                    h_start = out_i * stride
                    w_start = out_j * stride
                    # Recover the (kernel x kernel) gradient window.
                    # Only the cached argmax position receives the upstream gradient.
                    grad_window = scatter_grad_window(
                        d_out[n, c, out_i, out_j],
                        argmax[n, c, out_i, out_j],
                        kernel
                    )
                    # Add the gradient window back into the input gradient. 
                    # Use += because pooling windows may overlap when stride < kernel.
                    dx[n,c,h_start:h_start + kernel,w_start:w_start + kernel] += grad_window

    # Step 4: Return the input gradient.
    return dx

# Step 25 - relu_forward
import numpy as np

def relu_forward(x):
    # Step 1: Apply ReLU elementwise.
    # Positive values remain unchanged, while negative values become zero.
    out = np.maximum(0, x)
    # Step 2: Cache the original input.
    # The backward pass needs the input to determine which elements
    # were active (x > 0) and which were inactive (x <= 0).
    cache = {
        "x": x
    }
    # Step 3: Return the output and the cache.
    return out, cache

# Step 26 - relu_backward
import numpy as np

def relu_backward(d_out, cache):
    # Step 1: Retrieve the original input from the forward pass.
    x = cache["x"]
    # Step 2: Create a mask identifying the elements that were strictly positive during the forward pass.
    positive_mask = x > 0
    # Step 3: Propagate the upstream gradient only through the positive elements. 
    # Gradients for non-positive inputs become 0.
    dx = d_out * positive_mask
    # Step 4: Return the input gradient.
    return dx

# Step 27 - flatten_forward
import numpy as np

def flatten_forward(x):
    # Step 1: Cache the original input shape.
    # The backward pass needs this to restore the gradient to its original 4D layout.
    cache = {
        "x_shape": x.shape
    }
    # Step 2: Flatten each sample into a single row.
    # Original shape:
    # (N, C, H, W)
    # New shape:
    # (N, C * H * W)
    out = x.reshape(x.shape[0], -1)
    # Step 3: Return the flattened output and cache.
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # Step 1: Retrieve the original input shape saved during the forward pass.
    x_shape = cache["x_shape"]
    # Step 2: Reshape the upstream gradient back to the original
    # 4D feature map layout.
    # Forward:
    # (N, C, H, W) -> (N, C*H*W)
    # Backward:
    # (N, C*H*W) -> (N, C, H, W)
    dx = d_out.reshape(x_shape)
    # Step 3: Return the input gradient.
    return dx

# Step 29 - linear_forward
import numpy as np

def linear_forward(x, weights, bias):
    # Step 1: Compute the affine transformation.
    # x: (N, D_in)
    # weights: (D_in, D_out)
    # bias: (D_out,)
    # Output: (N, D_out)
    # NumPy automatically broadcasts the bias across all samples.
    out = x @ weights + bias
    # Step 2: Cache the tensors needed during the backward pass.
    cache = {
        "x": x,
        "weights": weights
    }
    # Step 3: Return the output and cache.
    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # Step 1: Retrieve the weight matrix saved during the forward pass.
    weights = cache["weights"]
    # Step 2: Backpropagate through the matrix multiplication.
    # Forward: Y = X @ W
    # Therefore: dX = dY @ W.T
    dx = d_out @ weights.T
    # Step 3: Return the input gradient.
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # Step 1: Backpropagate through the matrix multiplication.
    # Forward: Y = X @ W
    # Therefore: dW = X.T @ d_out
    dW = x.T @ dout
    # Step 2: Return the weight gradient.
    return dW

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # Step 1: Sum the upstream gradient across the batch dimension.
    # dout: (N, D_out)
    # Result: (D_out,)
    db = np.sum(dout, axis=0)
    # Step 2: Return the bias gradient.
    return db

# Step 33 - linear_backward
import numpy as np

def linear_backward(dout, cache):
    # Step 1: Compute the gradient with respect to the layer input.
    dx = linear_grad_input(dout, cache)
    # Step 2: Compute the gradient with respect to the weight matrix.
    # The input activations are retrieved from the forward-pass cache.
    dW = linear_grad_weights(cache["x"], dout)
    # Step 3: Compute the gradient with respect to the bias vector.
    db = linear_grad_bias(dout)
    # Step 4: Return all gradients in the order expected by the optimizer and the rest of the network.
    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # Step 1: Convert the raw logits into probabilities using a numerically stable softmax.
    probs = stable_softmax(logits)
    # Step 2: Compute the mean cross-entropy loss using the predicted probabilities and the ground-truth labels.
    loss = cross_entropy_loss(probs, y)
    # Step 3: Normalize -0.0 to +0.0 if necessary.
    if loss == 0.0:
        loss = 0.0
    return float(loss)

# Step 35 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(logits, y):
    # Step 1: Convert the logits into probabilities using the
    # numerically stable softmax implementation.
    probs = stable_softmax(logits)
    # Step 2: Convert the integer labels into one-hot vectors.
    # Shape: (N,) -> (N, C)
    targets = one_hot(y, probs.shape[1])
    # Step 3: Compute the fused softmax-cross-entropy gradient.
    # dL/dlogits = (probs - targets) / N
    dlogits = (probs - targets) / logits.shape[0]
    # Step 4 : Normalize signed zeros.
    dlogits[np.isclose(dlogits, 0.0)] = 0.0
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # Step 1: Move the parameter in the direction opposite to the gradient. The learning rate controls the step size.
    updated_param = param - lr * grad
    # Step 2: Return the updated parameter tensor.
    return updated_param

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # Step 1: Update the first moment estimate using an
    # exponential moving average (EMA) of the gradients.
    # m_t = beta_one * m_{t-1} + (1 - beta_one) * grad
    new_m = beta_one * m + (1.0 - beta_one) * grad
    # Step 2: Return the updated first moment estimate.
    return new_m

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # Step 1: Update the second moment estimate using an exponential moving average (EMA) of the squared gradients.
    # v_t = beta_two * v_{t-1} + (1 - beta_two) * (grad ** 2)
    new_v = beta_two * v + (1.0 - beta_two) * (grad ** 2)
    # Step 2: Return the updated second moment estimate.
    return new_v

# Step 39 - adam_bias_correct
import numpy as np

def adam_bias_correct(moment, beta, t):
    # Step 1: Undo the bias introduced by initializing the exponential moving average to zero.
    # corrected_moment = moment / (1 - beta ** t)
    corrected_moment = moment / (1.0 - beta ** t)
    # Step 2: Return the bias-corrected moment.
    return corrected_moment

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # Step 1: Compute the Adam parameter update.
    # Divide the bias-corrected first moment by the square root
    # of the bias-corrected second moment. The small epsilon
    # prevents division by zero and improves numerical stability.
    updated_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    # Step 2: Return the updated parameter array.
    # The original parameter tensor is left unchanged.
    return updated_param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # Step 1: Update the first moment (running average of gradients).
    new_m = adam_update_m(m, grad, beta_one)
    # Step 2: Update the second moment (running average of squared gradients).
    new_v = adam_update_v(v, grad, beta_two)
    # Step 3: Compute the bias-corrected first moment.
    m_hat = adam_bias_correct(new_m, beta_one, t)
    # Step 4: Compute the bias-corrected second moment.
    v_hat = adam_bias_correct(new_v, beta_two, t)
    # Step 5: Update the parameter using the bias-corrected moments.
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    # Step 6: Return the updated parameter together with the
    # new (uncorrected) moments for the next optimization step.
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
import numpy as np

def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # Step 1: Compute the fan-in for He initialization.
    # Each output activation depends on all input channels and
    # every element of the spatial kernel.
    fan_in = in_channels * kernel_size * kernel_size
    # Step 2: Initialize the convolution filters using He
    # initialization.
    W = he_init(
        shape=(out_channels,
               in_channels,
               kernel_size,
               kernel_size),
        fan_in=fan_in,
        seed=seed
    )
    # Step 3: Initialize the bias vector to zeros.
    b = init_zero_bias(out_channels)
    # Step 4: Return the layer parameters.
    return {
        "W": W,
        "b": b
    }

# Step 43 - init_linear_layer
import numpy as np

def init_linear_layer(in_features, out_features, seed=0):
    # Step 1: Compute the fan-in for He initialization.
    # Each output neuron receives all input features.
    fan_in = in_features
    # Step 2: Initialize the weight matrix using He initialization.
    W = he_init(
        shape=(in_features, out_features),
        fan_in=fan_in,
        seed=seed
    )
    # Step 3: Initialize the bias vector to zeros.
    b = init_zero_bias(out_features)
    # Step 4: Return the layer parameters.
    return {
        "W": W,
        "b": b
    }

# Step 44 - init_lenet
import numpy as np

def init_lenet(in_channels, num_classes, seed=0):
    # Step 1: Initialize the first convolution layer.
    # Weight shape: (6, in_channels, 5, 5)
    conv1 = init_conv_layer(
        out_channels=6,
        in_channels=in_channels,
        kernel_size=5,
        seed=seed
    )
    # Step 2: Initialize the second convolution layer.
    # Weight shape: (16, 6, 5, 5)
    conv2 = init_conv_layer(
        out_channels=16,
        in_channels=6,
        kernel_size=5,
        seed=seed + 1
    )
    # Step 3: After two 5×5 convolutions and two 2×2 pooling
    # layers on a 28×28 input:
    # 28 -> 24 -> 12 -> 8 -> 4
    # The flattened feature size is:
    # 16 × 4 × 4 = 256.
    fc1 = init_linear_layer(
        in_features=16 * 4 * 4,
        out_features=120,
        seed=seed + 2
    )
    # Step 4: Initialize the output classification layer.
    fc2 = init_linear_layer(
        in_features=120,
        out_features=num_classes,
        seed=seed + 3
    )
    # Step 5: Return all layer parameters.
    return {
        "conv1": conv1,
        "conv2": conv2,
        "fc1": fc1,
        "fc2": fc2
    }

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

