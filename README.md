# Face tracking with MDMs

This repository shows how to build a face tracker, following:

    Mnemonic Descent Method: A recurrent process applied for end-to-end face alignment
    G. Trigeorgis, P. Snape, M. A. Nicolaou, E. Antonakos, S. Zafeiriou.
    Proceedings of IEEE International Conference on Computer Vision & Pattern Recognition (CVPR'16).
    Las Vegas, NV, USA, June 2016.

## Requirements

To install most requirements:

pip install -r requirements.txt

To install OpenCV, the recommended option (through anaconda) is:

     conda install -c menpo opencv3=3.1.0 

To install tensorflow:

    ![Follow the instructions here](https://www.tensorflow.org/versions/r0.10/get_started/index.html)


## Training the face tracker

This face tracker is trained on the ![CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which is available for non commercial purposes.

We use the aligned images (crop + alignment of the face)

If you want to train the model from scratch, you will need to download the dataset from the link above.

You should configure the `data` folder as follows:

    ├── data
        ├── RawData
            ├── Anno
                ├──list_attr_celeba.txt  
                ├──list_landmarks_align_celeba.txt  
                └── list_landmarks_celeba.txt
            ├── Eval
                └──list_eval_partition.txt
            └── img_align_celeba

Unzip the aligned images in the `img_align_celeba` folder and you should be good to go !

### To build the dataset

In `src/model`:

    python make_dataset.py

### To start training

In `src/model`:

    python train.py

You may need to change some parameters like the batch size depending on your GPU.

### To monitor training:

In `models`:

    tensorboard --logdir=train

Then in your web browser, go to http://0.0.0.0:6006 to access tensorboard

## Testing the face tracker


Assuming you are sitting in front of your webcam:

In `src/model`:

    python tracker.py

You can modify the choice of model by specifying your model checkpoint at the beginning of the script

    tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                               """If specified, restore this pretrained model """
                               """before beginning any training.""")

By default, it is set to the pre-trained model that you can find in `models`