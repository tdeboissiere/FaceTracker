from datetime import datetime
import models
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import h5py
import matplotlib.pylab as plt
import utils
from tqdm import tqdm as tqdm


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 1E-3,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', '../models/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '../models/train/model.ckpt-11',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def sample_perturbation(gt_pts, mean_shape):
    """
    Sample perturbations
    """

    # compute rigid parameters by aligning mean shape with gt_pts
    g_s,g_r,g_tx,g_ty = utils.CalcSimT(np.ravel(mean_shape, order="F"),
                                       np.ravel(gt_pts, order='F'))
    g_s = np.sqrt(g_s**2 + g_r**2)
    g_r = 0

    # set perturbation limits
    perturb_scale, perturb_tx, perturb_ty = 0.15, 30, 30
    perturb_scale_small, perturb_tx_small, perturb_ty_small = 0.1, 10, 10

    # sample perturbation
    r = np.random.uniform()
    if(r < 0.2):
        temp_s = np.random.uniform(g_s - perturb_scale_small, g_s + perturb_scale_small)
        temp_r = g_r
        temp_tx = np.random.uniform(g_tx - perturb_tx_small, g_tx + perturb_tx_small)
        temp_ty = np.random.uniform(g_ty - perturb_ty_small, g_ty + perturb_ty_small)
    else:
        temp_s = np.random.uniform(g_s - perturb_scale, g_s + perturb_scale)
        temp_r = g_r
        temp_tx = np.random.uniform(g_tx - perturb_tx, g_tx + perturb_tx)
        temp_ty = np.random.uniform(g_ty - perturb_ty, g_ty + perturb_ty)

    # apply perturbations to the reference shape
    perturbed_pts = utils.SimT(np.ravel(mean_shape, order="F"), temp_s, temp_r, temp_tx, temp_ty)
    perturbed_pts = np.reshape(perturbed_pts, gt_pts.shape, order='F')

    return perturbed_pts


def get_perturbation_statistics():

    with h5py.File("../data/training_celeba_FaceTracker.h5") as hf:
        tr_data = hf["training_data"][:]
        tr_landmarks = hf["training_landmarks"][:]
        mean_landmarks = hf["mean_landmarks"][:]

    with h5py.File("../data/validation_celeba_FaceTracker.h5") as hf:
        val_data = hf["validation_data"][:]
        val_landmarks = hf["validation_landmarks"][:]

    with h5py.File("../data/test_celeba_FaceTracker.h5") as hf:
        test_data = hf["test_data"][:]
        test_landmarks = hf["test_landmarks"][:]

    # Stack data and landmarks
    data = np.vstack([tr_data, val_data, test_data])
    landmarks = np.vstack([tr_landmarks, val_landmarks, test_landmarks])

    # Now sample perturbations
    delta = []
    for i in tqdm(range(1000000)):
        idx = np.random.randint(0, len(data))
        ldm = landmarks[idx]
        perturbed_ldm = sample_perturbation(ldm, mean_landmarks)
        delta.append(ldm - perturbed_ldm)
    delta = np.array(delta)

    delta_mean = np.mean(delta, 0)
    delta_std = np.std(delta, 0)

    np.save("../data/delta_mean.npy", delta_mean)
    np.save("../data/delta_std.npy", delta_std)

    return delta_mean, delta_std


def train(scope=''):
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = 100
        num_epochs_per_decay = 5
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        # Set the number of preprocessing threads
        num_preprocess_threads = FLAGS.num_preprocess_threads

        with h5py.File("../data/training_celeba_FaceTracker.h5") as hf:
            _images_train = hf["training_data"][:]
            _landmarks_train = hf["training_landmarks"][:]
            mean_landmarks = hf["mean_landmarks"][:]

        with h5py.File("../data/validation_celeba_FaceTracker.h5") as hf:
            _images_val = hf["validation_data"][:]
            _landmarks_val = hf["validation_landmarks"][:]

        # Load the mean vector and std of (true_landmark - perturbed_landmark)
        try:
            delta_mean = np.load("../data/delta_mean.npy")
            delta_std = np.load("../data/delta_std.npy")
        except:
            delta_mean, delta_std = get_perturbation_statistics()

        image_shape = _images_train[0].shape
        lms_shape = _landmarks_train[0].shape

        def get_random_sample():
            idx = np.random.randint(0, len(_images_train))
            shape = _landmarks_train[idx].astype("float32")
            initial_shape = sample_perturbation(shape, mean_landmarks).astype("float32")
            # plt.imshow(_images_train[idx][:, :, 0], cmap="gray")
            # plt.scatter(shape[:, 0], shape[:, 1], c="g")
            # plt.scatter(initial_shape[:, 0], initial_shape[:, 1], c="r")
            # plt.show()
            # plt.clf()
            # plt.close()
            return _images_train[idx].astype("float32"), shape, initial_shape

        image, shape, initial_shape = tf.py_func(get_random_sample, [],
                                                 [tf.float32, tf.float32, tf.float32], name="random_sample_train")
        image.set_shape(image_shape)
        shape.set_shape(lms_shape)
        initial_shape.set_shape(lms_shape)

        images, lms, inits = tf.train.batch([image, shape, initial_shape],
                                            FLAGS.batch_size,
                                            dynamic_pad=False,
                                            capacity=1000,
                                            enqueue_many=False,
                                            num_threads=num_preprocess_threads,
                                            name='train_img_batch')

        def get_random_sample_val():
            idx = np.random.randint(0, len(_images_val))
            shape = _landmarks_val[idx].astype("float32")
            initial_shape = sample_perturbation(shape, mean_landmarks).astype("float32")
            return _images_val[idx].astype("float32"), shape, initial_shape

        image_val, shape_val, initial_shape_val = tf.py_func(get_random_sample_val, [],
                                                             [tf.float32, tf.float32, tf.float32],
                                                             name="random_sample_val")
        image_val.set_shape(image_shape)
        shape_val.set_shape(lms_shape)
        initial_shape_val.set_shape(lms_shape)

        images_val, lms_val, inits_val = tf.train.batch([image_val, shape_val, initial_shape_val],
                                                        FLAGS.batch_size,
                                                        dynamic_pad=False,
                                                        capacity=1000,
                                                        enqueue_many=False,
                                                        num_threads=num_preprocess_threads,
                                                        name='val_img_batch')

        print('Defining model...')
        with tf.device(FLAGS.train_device):
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            with tf.variable_scope("scopernn") as scopernn:
                predictions, dxs, _ = models.model(images, inits, is_training=True)
                scopernn.reuse_variables()
                predictions_val, dxs_val, _ = models.model(images_val, inits_val, is_training=False)

            total_loss_train = 0
            total_loss_val = 0
            list_train_loss, list_val_loss = [], []

            loss_weights = [1, 1, 1, 1]
            with tf.name_scope("Error_train"):
                for i, dx in enumerate(dxs):
                    loss_norm, loss = models.normalized_rmse(inits, dx, lms, delta_mean, delta_std)
                    tf.histogram_summary('errors', loss)
                    list_train_loss.append(loss)
                    total_loss_train += loss_norm * loss_weights[i]
                    summaries.append(tf.scalar_summary('losses_train/step_{}'.format(i),
                                                       loss))
            with tf.name_scope("Error_val"):
                for i, dx in enumerate(dxs_val):
                    loss_norm_val, loss_val = models.normalized_rmse(inits_val, dx, lms_val, delta_mean, delta_std)
                    tf.histogram_summary('errors', loss_val)
                    list_val_loss.append(loss_val)
                    total_loss_val += loss_norm_val * loss_weights[i]
                    summaries.append(tf.scalar_summary('losses_val/step_{}'.format(i),
                                                       loss_val))

            # Calculate the gradients for the batch of data
            grads = opt.compute_gradients(total_loss_train)

        summaries.append(tf.scalar_summary('losses/total_train', total_loss_train))
        summaries.append(tf.scalar_summary('losses/total_val', total_loss_val))

        gt_images_val, = tf.py_func(utils.batch_draw_landmarks_green, [images_val, lms_val],
                                    [tf.float32], name="gt_img_visu")
        init_images_val, = tf.py_func(utils.batch_draw_landmarks_red, [images_val, inits_val],
                                      [tf.float32], name="init_img_visu")
        pred_images_val, = tf.py_func(utils.batch_draw_landmarks_green,
                                      [images_val, predictions_val], [tf.float32], name="pred_img_visu")

        summary = tf.image_summary('images_val',
                                   tf.concat(2, [gt_images_val, init_images_val, pred_images_val]),
                                   max_images=8)
        summaries.append(tf.histogram_summary('dx_train', predictions - inits))
        summaries.append(tf.histogram_summary('dx_val', predictions_val - inits_val))

        summaries.append(summary)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope)

        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.histogram_summary(var.op.name +
                                                      '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        print('Starting evaluation...')
        nb_epoch = 200
        n_step_per_epoch = 500
        for ep in range(nb_epoch):
            for step in range(n_step_per_epoch):
                imgs, gts, ini, preds = sess.run([images_val, lms_val, inits_val, predictions_val])

                for k in range(128):
                    plt.imshow(imgs[k, :, :, 0], cmap="gray")
                    plt.scatter(gts[k, :, 0], gts[k, :, 1], color="green")
                    plt.scatter(ini[k, :, 0], ini[k, :, 1], color="red")
                    plt.scatter(preds[k, :, 0], preds[k, :, 1], color="blue")
                    plt.show()
                    plt.clf()
                    plt.close()
                raw_input()

if __name__ == '__main__':
    train()
