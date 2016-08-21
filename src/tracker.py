import numpy as np
import cPickle as pickle
import cv2
import tensorflow as tf
import h5py
import sys
from datetime import datetime
from tqdm import tqdm
sys.path.append("../model")
import models
import slim
import utils


# set params
visualize = 1
webcam = True
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt_tree.xml')

# main code
nlandmarks = 5

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 1E-3,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 1, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', '../../models/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '../models/train/model.ckpt-17',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('datasets', ':'.join(
    ('databases/lfpw/trainset/*.png', 'databases/afw/*.jpg',
     'databases/helen/trainset/*.jpg')),
    """Directory where to write event logs """
    """and checkpoint.""")
MOVING_AVERAGE_DECAY = 0.999


def format_img(gray, init_shape):

    leye = init_shape[0]
    reye = init_shape[1]
    eye_dist = reye[0] - leye[0]
    print eye_dist
    scale_factor = 44. / eye_dist  # we want ~ 44 pixels for the inter eye distance

    # Use the best (according to opencv) interpolation method
    # depending on the magnitude of the scale factor
    if scale_factor < 1:
        itp = cv2.INTER_AREA
    else:
        itp = cv2.INTER_CUBIC
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=itp)

    init_shape *= scale_factor

    leye_x, leye_y = init_shape[0]

    # Get image new shape
    h, w = gray.shape
    # Now slice the image total size = 128. Make sure we dont pad with zeros
    # to do so: rework boundaries when needed
    min_y, max_y = int(leye_y) - 42, int(leye_y) + 86
    if min_y < 0:
        min_y = 0
        max_y = 128
    if max_y > h:
        max_y = h
        min_y = h - 128
    min_x, max_x = int(leye_x) - 42, int(leye_x) + 86
    if min_x < 0:
        min_x = 0
        max_x = 128
    if max_x > w:
        max_x = w
        min_x = w - 128
    init_shape[:, 0] -= min_x
    init_shape[:, 1] -= min_y
    cropped_gray = gray[min_y: max_y, min_x: max_x]
    assert cropped_gray.shape == (128, 128)
    cropped_gray = cropped_gray.reshape(128, 128, 1).astype(np.uint8)
    init_shape = init_shape.reshape((5, 2))

    return cropped_gray, init_shape


def compute_init_shape(mean_landmarks, faces):
    """
    Utility to recenter / rescale the mean shape to the opencv face window
    """

    ref_ldm = np.reshape(mean_landmarks, (5, 2), order='F')
    x,y,w,h = faces[0]

    # The translation and scale coefficients were found experimentally
    scale = 0.30 * w / (max(ref_ldm[:,0]) - min(ref_ldm[:,0]))
    tx = x + 0.05 * w
    ty = y + 0.05 * h

    init_shape = ref_ldm.copy()
    init_shape = scale * init_shape
    init_shape[:,0] = init_shape[:,0] + tx
    init_shape[:,1] = init_shape[:,1] + ty

    return init_shape


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


def generate_overlay(frame, shape, init_shape):

    c = (0,255,0)
    c2 = (0,0,255)

    # points
    for j in xrange(shape.shape[0]):
        cv2.circle(frame, (int(shape[j, 0]), int(shape[j, 1])), 2, c, -1)
    for j in xrange(init_shape.shape[0]):
        cv2.circle(frame, (int(init_shape[j, 0]), int(init_shape[j, 1])), 2, c2, -1)


def main():
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
            _images_train = hf["training_data"][:10]
            _landmarks_train = hf["training_landmarks"][:10]
            mean_landmarks = hf["mean_landmarks"][:]

        with h5py.File("../data/validation_celeba_FaceTracker.h5") as hf:
            _images_val = hf["validation_data"][:10]
            _landmarks_val = hf["validation_landmarks"][:10]

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
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, "")
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
                                              "")

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

        #################
        # APP
        #################
        cap = cv2.VideoCapture(0)

        mode = 0  # detect
        shape = []
        init_shape = []
        print '\n\nPRESS q/Q to QUIT\n'

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret is True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect face with Haar cascade
                if mode == 0:
                    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
                    if len(faces) == 0:
                        continue

                # face detection succesfull. Start Tracking!

                if mode == 0:
                    init_shape = compute_init_shape(mean_landmarks, faces)
                else:
                    # Need to realign init_shape with mean shape
                    a,b,tx,ty = utils.CalcSimT(np.ravel(mean_landmarks, order='F'), init_shape.ravel('F'))
                    init_shape = utils.SimT(np.ravel(mean_landmarks, order='F'), a, b, tx, ty)
                    init_shape = np.reshape(init_shape, (5, 2), order='F')

                leyex, leyey = init_shape[0]
                reyex, reyey = init_shape[1]

                ax, bx = 44. / (reyex - leyex), 44. * (1 - leyex / (reyex - leyex))
                ay, by = 44. / (reyex - leyex), 44. * (1 - leyey / (reyex - leyex))

                # # # # Format image to 128 x 128 and rescale the init shape
                gray_cropped, init_cropped = format_img(gray, init_shape.copy())
                gcc = gray_cropped.copy()

                gray_cropped = gray_cropped.reshape((1, 128, 128, 1)).astype(np.float32)
                init_cropped = init_cropped.reshape((1, 5, 2))

                # import matplotlib.pylab as plt
                # # # img = cv2.imread("000091.jpg", 0)
                # # # img = img[40:170, 40:150]
                # # # img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                # # # gray_cropped = img.reshape((1, 128, 128, 1)) / 255.
                # bla = _images_train[4].reshape((1, 128, 128, 1))

                # plt.imshow(bla[0, :, :, 0], cmap="gray")
                # plt.scatter(init_cropped[0, :, 0], init_cropped[0, :, 1])
                # plt.scatter(_landmarks_train[4, :, 0], _landmarks_train[4, :, 1], color="green", s=40)
                # preds = sess.run(predictions_val, feed_dict={images_val:bla,
                #                                              inits_val:init_cropped})
                # plt.scatter(preds[0, :, 0], preds[0, :, 1], color="red")
                # plt.show()
                # raw_input()

                preds = sess.run(predictions_val, feed_dict={images_val:gray_cropped,
                                                             inits_val:init_cropped})
                preds = preds[0]
                # # # Convert preds to the big image scale
                preds[:,0] = (preds[:, 0] - bx) / (ax)
                preds[:,1] = (preds[:, 1] - by) / (ay)

                mode = 1

                # if len(faces) != 0:
                #     for (x, y, w, h) in faces:
                #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # if init_shape.shape[0] > 0:
                #     for k in range(5):
                #         cv2.circle(frame,(int(init_shape[k,0]),int(init_shape[k,1])),2,(0,0,255),-1)

                # if init_cropped != []:
                #     for k in range(5):
                #         cv2.circle(gcc,(int(init_cropped[0][k,0]),int(init_cropped[0][k,1])),2,(0,0,255),-1)
                #         # cv2.circle(gcc,(int(preds[k,0]),int(preds[k,1])),2,(0,0,255),-1)

            # if len(faces) == 0:
            #     return shape, head_pose, score, mode, track_time

                if mode != 0:
                    generate_overlay(frame,
                                     preds,
                                     init_shape)
                    cv2.imshow('Face Tracker (q/Q: Quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    # for next frame
                    init_shape = preds
                else:
                    cv2.imshow('Face Tracker (q/Q: Quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            else:

                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    main()
