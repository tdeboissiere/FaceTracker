import slim
import tensorflow as tf

from slim import ops
from slim import scopes

import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import h5py


def normalized_rmse(inits, dx_pred, gt_truth, dx_mean, dx_std):
    dx_pred_shape = dx_pred.get_shape().as_list()
    gt_truth_shape = gt_truth.get_shape().as_list()
    num_lm = gt_truth_shape[1]
    assert dx_pred_shape == gt_truth_shape, "Conflicting predicted and ground truth shapes"

    pred = inits + dx_pred
    pred = tf.reshape(pred, (dx_pred_shape[0], num_lm * 2, ))
    gt_truth = tf.reshape(gt_truth, (dx_pred_shape[0], num_lm * 2, ))
    inits = tf.reshape(inits, (dx_pred_shape[0], num_lm * 2, ))
    dx_pred = tf.reshape(dx_pred, (dx_pred_shape[0], num_lm * 2, ))

    # Compute 2 rmse: with and without normalisation
    # RMSE with normalisation is used to compute the gradients
    dx_gt = gt_truth - inits
    dx_gt -= np.ravel(dx_mean)
    dx_gt /= np.ravel(dx_std)

    dx_pred -= np.ravel(dx_mean)
    dx_pred /= np.ravel(dx_std)

    rmse_norm = tf.square(dx_pred - dx_gt)
    rmse_norm = tf.sqrt(tf.reduce_mean(rmse_norm, 1))
    rmse_norm = tf.reduce_mean(rmse_norm)

    rmse = tf.square(pred - gt_truth)
    rmse = tf.sqrt(tf.reduce_mean(rmse, 1))
    rmse = tf.reduce_mean(rmse)
    return rmse_norm, rmse


def get_central_crop(images, box=(6, 6)):
    _, w, h, _ = images.get_shape().as_list()

    half_box = (box[0] / 2., box[1] / 2.)

    a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
    b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))

    return images[:, a, b, :]


def extract_patches(imgs, landmarks, patch_shape):
    """ Extracts patches from an image.

    Args:
        imgs: a numpy array of dimensions [width, height, channels]
        landmarks: a numpy array of dimensions [num_patches, 2]
        sampling_grid: (patch_width, patch_height, 2)

    Returns:
        a numpy array [num_patches, width, height, channels]
    """

    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)

    list_patches = []

    for i in range(imgs.shape[0]):

        img, ldm = imgs[i], landmarks[i]
        img = img.transpose(2, 0, 1)

        max_x = img.shape[-2] - 1
        max_y = img.shape[-1] - 1

        patch_grid = (sampling_grid[None, :, :, :] + ldm[:, None, None, :]
                      ).astype('int32')

        X = patch_grid[:, :, :, 0].clip(0, max_x)
        Y = patch_grid[:, :, :, 1].clip(0, max_y)

        patches = img[:, Y, X].transpose(1, 3, 2, 0)
        list_patches.append(patches)

        # # Plot for debugging
        # plt.figure()
        # plt.imshow(img[0, :, :], cmap="gray")
        # plt.scatter(ldm[:, 0], ldm[:, 1])

        # gs = gridspec.GridSpec(5, 1)
        # fig = plt.figure(figsize=(15, 15))
        # for i in range(5):
        #     ax = plt.subplot(gs[i])
        #     ax.imshow(patches[i, :, :, 0], cmap="gray")
        # gs.tight_layout(fig)
        # plt.show()

    return np.array(list_patches).astype(np.float32)


def conv_model(inputs, is_training=True, scope=''):

    # summaries or losses.
    net = {}

    with tf.op_scope([inputs], scope, 'mdm_conv'):
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1'] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1')
                net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
                net['conv_2'] = ops.conv2d(net['pool_1'], 32, [3, 3], scope='conv_2')
                net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])

                crop_size = net['pool_2'].get_shape().as_list()[1:3]
                net['conv_2_cropped'] = get_central_crop(net['conv_2'], box=crop_size)

                net['concat'] = tf.concat(3, [net['conv_2_cropped'], net['pool_2']])
    return net


def model(images, inits, num_iterations=3, num_patches=5, patch_shape=(24, 24), num_channels=1, is_training=True):
    batch_size = images.get_shape().as_list()[0]
    hidden_state = tf.zeros((batch_size, 512))
    dx = tf.zeros((batch_size, num_patches, 2))
    images /= 255.
    endpoints = {}
    dxs = []

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = tf.py_func(extract_patches, [images, inits + dx, tf.constant(patch_shape)], [tf.float32])[0]
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))

        endpoints['patches'] = patches
        with tf.variable_scope('convnet', reuse=step > 0):
            net = conv_model(patches)
            ims = net['concat']

        ims = tf.reshape(ims, (batch_size, -1))

        with tf.variable_scope('rnn', reuse=step > 0):
            hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
            hidden_drop = slim.ops.dropout(hidden_state, 0, scope='drop', is_training=is_training)
            prediction = slim.ops.fc(hidden_drop, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)

    return inits + dx, dxs, endpoints


if __name__ == '__main__':

    with h5py.File("../data/validation_celeba_FaceTracker.h5") as hf:
        _images_val = hf["validation_data"][:]
        _landmarks_val = hf["validation_landmarks"][:]

        extract_patches(_images_val[:10], _landmarks_val[:10], (64, 64))
