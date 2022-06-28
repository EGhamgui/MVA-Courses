import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from imageio import imread
import numpy as np
import cv2
import os
from helper import mask_labels


FRAME_SIZE = 96
NB_IMGS_AUGMENTED = 50  


def default_preprocess_image(image):
    """
    Parameters
    ----------
    image : array
        (214, 214) array representing a grayscale image
    
    Returns
    -------
        (96, 96) preprocessed image
    """
    output_frame_size = 96   # do not change the output frame size!
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    im = cv2.resize(image, (output_frame_size, output_frame_size)).astype(np.float)
    im -= im.mean()
    im /= im.max()
    image = im
    return image


def plot_image_data(images):
    plt.figure(1, figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(images[i].reshape(FRAME_SIZE, FRAME_SIZE))
        r='{:d}'.format(i+1)
        if i < 10:
            plt.title(r)
    plt.show()


def plot_image_data_augmented(images):
    plt.figure(1, figsize=(15, 10))
    for i in range(10 * NB_IMGS_AUGMENTED):
        plt.subplot(NB_IMGS_AUGMENTED,10,i+1)
        plt.axis('off')
        plt.imshow(images[i].reshape(FRAME_SIZE, FRAME_SIZE))
        r='{:d}'.format(i+1)
        if i < 10:
            plt.title(r)
    plt.show()


def load_image_data(preprocess_fn=default_preprocess_image):
    """
    Load image dataset.

    Returns
    -------
    images : array
        Data matrix, shape: (n_images, frame_size**2)
    labels : array
        True labels
    masked_labels: array
        Masked labels, with 4 revealed labels per person.
    """
    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    # Loading images
    images = np.zeros((100, FRAME_SIZE ** 2))
    labels = np.zeros(100, dtype=np.uint32)

    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = preprocess_fn(gray_face)

            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
    select 4 random labels per person and reveal them  
    masked_labels: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    masked_labels = mask_labels(labels, 4, per_class=True)

    return images, labels, masked_labels


def load_image_data_augmented(preprocess_fn=default_preprocess_image, number_masked = 4, num_per = 10):
    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    # Loading images
    images = np.zeros((num_per * NB_IMGS_AUGMENTED, FRAME_SIZE ** 2))
    labels = np.zeros(num_per * NB_IMGS_AUGMENTED, dtype=np.uint32)

    for i in np.arange(num_per):
        imgdir = "data/extended_dataset/%d" % i
        imgfns = os.listdir(imgdir)
        for j, imgfn in enumerate(np.random.choice(imgfns, size=NB_IMGS_AUGMENTED)):
            im = imread("{}/{}".format(imgdir, imgfn))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = preprocess_fn(gray_face)

            # resize the face and reshape it to a row vector, record labels
            images[j * num_per + i] = gray_face.reshape((-1))
            labels[j * num_per + i] = i + 1

    """
    select 4 random labels per person and reveal them  
    masked_labels: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    masked_labels = mask_labels(labels, number_masked, per_class=True)

    return images, labels, masked_labels


if __name__=='__main__':
    images, labels, masked_labels = load_image_data_augmented()
