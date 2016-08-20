import os
import h5py
import cv2
import numpy as np
import parmap
import cPickle as pickle
import matplotlib.pylab as plt
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm as tqdm


def format_image(img_path, d_land):
    """
    Load img with opencv, crop around landmarks and reshape
    return img and landmarks
    """

    img = cv2.imread(img_path, 0)
    d_landmarks = d_land[os.path.basename(img_path)]

    reyey = d_landmarks["righteye_y"]
    leyex, leyey = d_landmarks["lefteye_x"], d_landmarks["lefteye_y"]
    lmouthy = d_landmarks["leftmouth_y"]

    # Bounding box for the close up of the face
    # (a,b,c = topleft, top right, bottom left)
    a = leyex - 40, leyey - 40
    b = reyey + 40, leyey - 40
    c = leyex - 40, lmouthy + 30

    # We want to resize the face close up to 128 x 128
    fy = 128. / (c[1] - a[1])
    fx = 128. / (b[0] - a[0])

    # Resize the image
    img_cropped = cv2.resize(img[a[1]:c[1], a[0]:b[0]], None, fx=fx, fy=fy)

    # We also need to transform the landmark locations
    new_landmarks = []
    for lm in ["lefteye", "righteye", "nose", "leftmouth", "rightmouth"]:
        new_landmarks.append([d_landmarks["%s_x" % lm], d_landmarks["%s_y" % lm]])
    new_landmarks = np.array(new_landmarks).astype(np.float32)
    new_landmarks[:, 0] -= a[0]
    new_landmarks[:, 1] -= a[1]
    new_landmarks[:, 0] *= fx
    new_landmarks[:, 1] *= fy

    return img_cropped.reshape((1, 128, 128, 1)), new_landmarks.reshape((1, 5, 2))


def build_HDF5(dset_type):
    """
    Gather the data in a single HDF5 file.
    """

    # Define Raw Dir
    raw_dir = os.path.expanduser(os.environ.get("RAW_DIR"))
    # Define  directory where processed files will be stored
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Read landmarks file
    # Build it if it does not exist
    d_land = {}
    try:
        with open(data_dir + "d_land_align.pickle", "r") as fd:
            print "Loading d_land"
            d_land = pickle.load(fd)
            list_land = sorted(d_land[d_land.keys()[0]].keys())
    except:
        with open(raw_dir + "Anno/list_landmarks_align_celeba.txt", "r") as f:
            lines = f.readlines()
            list_land = lines[1].rstrip().split()
            for celeb in lines[2:]:
                celeb = celeb.rstrip().split()
                img = celeb[0]
                attrs = map(int, celeb[1:])
                d_land[img] = {att:val for att,val in zip(list_land, attrs)}
            list_land = sorted(d_land[d_land.keys()[0]].keys())
        with open(data_dir + "d_land_align.pickle", "w") as fd:
            pickle.dump(d_land, fd)

    # Read evaluation file, build it if it does not exist
    # In evaluation status, "0" represents training image, "1" represents
    # validation image, "2" represents testing image;
    d_eval = {}
    try:
        with open(data_dir + "d_eval.pickle", "r") as fd:
            print "Loading d_eval"
            d_eval = pickle.load(fd)
    except:
        with open(raw_dir + "Eval/list_eval_partition.txt", "r") as f:
            lines = f.readlines()
            for celeb in lines:
                celeb = celeb.rstrip().split()
                img = celeb[0]
                attrs = int(celeb[1])
                d_eval[img] = attrs
        with open(data_dir + "d_eval.pickle", "w") as fd:
            pickle.dump(d_eval, fd)

    # Get the list of jpg files
    list_img = []
    if dset_type == "training":
        for img in d_eval.keys():
            if d_eval[img] == 0:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))
    elif dset_type == "validation":
        for img in d_eval.keys():
            if d_eval[img] == 1:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))
    elif dset_type == "test":
        for img in d_eval.keys():
            if d_eval[img] == 2:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))

    # Shuffle images
    np.random.seed(20)
    p = np.random.permutation(len(list_img))
    list_img = np.array(list_img)[p]

    # Put train data in HDF5
    hdf5_file = os.path.join(data_dir, "%s_celeba_FaceTracker.h5" % dset_type)
    with h5py.File(hdf5_file, "w") as hfw:

        data = hfw.create_dataset("%s_data" % dset_type,
                                  (0, 128, 128, 1),
                                  maxshape=(None, 128, 128, 1),
                                  dtype=np.uint8)

        landmarks = hfw.create_dataset("%s_landmarks" % dset_type,
                                       (0, 5, 2),
                                       maxshape=(None, 5, 2),
                                       dtype=np.float32)

        num_files = len(list_img)
        chunk_size = 50
        num_chunks = num_files / chunk_size
        arr_chunks = np.array_split(np.arange(num_files), num_chunks)

        for chunk_idx in tqdm(arr_chunks):

            list_img_path = list_img[chunk_idx].tolist()
            output = parmap.map(format_image, list_img_path, d_land, parallel=False)

            arr_img = np.vstack([o[0] for o in output if o[0].shape[0] > 0])
            arr_landmarks = np.vstack([o[1] for o in output if o[0].shape[0] > 0])

            # Resize HDF5 dataset
            data.resize(data.shape[0] + arr_img.shape[0], axis=0)
            landmarks.resize(landmarks.shape[0] + arr_img.shape[0], axis=0)

            data[-arr_img.shape[0]:] = arr_img.astype(np.uint8)
            landmarks[-arr_img.shape[0]:] = arr_landmarks.astype(np.float32)


def check_HDF5(dset_type):
    """
    Plot images with landmarks to check the processing
    """

    # Get processed data directory
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    # Get hdf5 file
    hdf5_file = os.path.join(data_dir, "%s_celeba_FaceTracker.h5" % dset_type)

    with h5py.File(hdf5_file, "r") as hf:
        data = hf["%s_data" % dset_type]
        landmarks = hf["%s_landmarks" % dset_type]
        mean_landmarks = hf["mean_landmarks"]
        for i in range(data.shape[0]):
            plt.imshow(data[i, :, :, 0], cmap="gray")
            plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], color="green", s=40)
            plt.scatter(mean_landmarks[:, 0], mean_landmarks[:, 1], color="red", s=40)
            plt.show()
            plt.clf()
            plt.close()


def get_landmark_stats():
    """
    Utility to check the landmarks' distribution
    """

    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    from tqdm import tqdm as tqdm

    with open(data_dir + "d_land_align.pickle", "r") as fd:
        print "Loading d_land"
        d_land = pickle.load(fd)
        l = []
        for key in tqdm(d_land.keys()):
            d = d_land[key]["righteye_x"] - d_land[key]["lefteye_x"]
            l.append(d)
        plt.hist(l, bins=100)
        plt.show()


def get_mean_landmarks():
    """
    Gather the data and compute the mean shape = the mean position of the landmarks
    """
    list_landmarks = []

    # Get processed data directory
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    # Loop over datasets
    for dset_type in ["training", "validation", "test"]:
        hdf5_file = os.path.join(data_dir, "%s_celeba_FaceTracker.h5" % dset_type)
        with h5py.File(hdf5_file, "r") as hf:
            list_landmarks.append(hf["%s_landmarks" % dset_type][:])

    # Compute mean landmarks
    landmarks = np.vstack(list_landmarks)
    mean_landmarks = np.mean(landmarks, 0)

    # Reloop over datasets and add a mean_landmarks dataset
    for dset_type in ["training", "validation", "test"]:

        hdf5_file = os.path.join(data_dir, "%s_celeba_FaceTracker.h5" % dset_type)
        with h5py.File(hdf5_file, "a") as hf:
            hf.create_dataset("mean_landmarks", data=mean_landmarks)

if __name__ == '__main__':

    load_dotenv(find_dotenv())

    # Check the env variables exist
    raw_msg = "Set your raw data absolute path in the .env file at project root"
    data_msg = "Set your processed data absolute path in the .env file at project root"
    assert "RAW_DIR" in os.environ, raw_msg
    assert "DATA_DIR" in os.environ, data_msg

    # get_landmark_stats()
    # build_HDF5("training")
    # build_HDF5("validation")
    # build_HDF5("test")
    check_HDF5("validation")
    # get_mean_landmarks()
