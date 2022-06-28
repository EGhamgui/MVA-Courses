import cv2 as cv
import os
import numpy as np
from scipy.spatial import distance


face_haar_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_haar_cascade = cv.CascadeClassifier("data/haarcascade_eye.xml")


def online_face_recognition(profile_names,
                            IncrementalKCenters,
                            n_pictures=15,
                            video_filename=None):
    """
    Run online face recognition.

    Parameters
    ----------
    profile_names : list
        List of user names used in create_user_profile()
    IncrementalKCenters : class
        Class implementing Incremental k-centers
    n_pictures : int
        Number of (labeled) pictures to use for each user_name
    video_filename : str
        .mp4 video file. If None, read from camera
    """
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [np.ones(p.shape[0]) * (i + 1)]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int)
    #  Generate model
    model = IncrementalKCenters(faces, labels, label_names)
    
    # Start camera
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            # look for eye classifier
            local_image = img[y:(y + y_range), x:(x + x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]),
                             (0, 0, 255), 2)
                continue
            # select face
            local_image = grey_image[y:(y + y_range), x:(x + x_range)]
            x_t = preprocess_face(local_image)

            """
            Centroids are updated here
            """
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])

            """
            Hard HFS solution is computed here
            """
            label_scores = model.online_ssl_compute_solution()
            scores = [ll[1] for ll in label_scores]
            labels = [ll[0] for ll in label_scores]
            sorted_label_indices = np.argsort(scores)

            """
            Show results
            """
            for ii, ll_idx in enumerate(sorted_label_indices):
                label = labels[ll_idx]
                score = scores[ll_idx]
                if label not in label_names:
                    color = (100, 100, 100)
                else:
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][ll_idx % 3]
                txt = label + "  " + ('%.4f' % score)
                cv.putText(img, txt, (p1[0], p1[1] - 5 - 10 * ii), cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                0.5 + 0.5 * (ii == len(scores) - 1), color)

            cv.rectangle(img, p1, p2, color, 2)
        cv.putText(img, "Face recognition: [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]:
            break
        if key == ord('s'):
            # Save face
            print('saved')
            # cv.imwrite("frame.png", img)
            if not os.path.exists('results'):
                os.makedirs('results')
            image_name = os.path.join('results', 'frame.png')
            cv.imwrite(image_name, img)
            print("Image saved at", image_name)

            ## cv.waitKey(1)
    cv.destroyAllWindows()



def create_user_profile(user_name,
                        faces_path="data/",
                        video_filename=None):
    """
    Uses the camera to collect data.
    
    Parameters
    ----------
    user_name : str
        Name that identifies the person/face
    faces_path : str
        Where to store the images
    video_filename : str
        .mp4 video file. If None, read from camera
    """
    # Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    image_count = 0
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        print("New profile created at path", profile_path)
    else:
        image_count = len(os.listdir(profile_path))
        print("Profile found with", image_count, "images.")
    # Launch video capture
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        if len(box) > 0:
            box_surface = box[:, 2] * box[:, 3]
            index = box_surface.argmax()
            b0 = box[index]
            cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]), (0, 255, 0),
                         2)
        cv.putText(img, f"Create profile ({user_name}): [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]: break  # esc or e to quit
        if key == ord('s'):
            ## Save face
            if len(box) > 0:
                x, y = b0[0], b0[1]
                x_range, y_range = b0[2], b0[3]
                image_count = image_count + 1
                image_name = os.path.join(profile_path, "img_" + str(image_count) + ".bmp")
                img_to_save = img[y:(y + y_range), x:(x + x_range)]
                cv.imwrite(image_name, img_to_save)
                print("Image", image_count, "saved at", image_name)
    cv.destroyAllWindows()
    return


def load_profile(user_name, faces_path="data/"):
    """
    Loads the data associated to user_name.

    Returns an array of shape (number_of_images, n_pixels)
    """
    assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    if not os.path.exists(profile_path):
        raise Exception("Profile not found")
    image_count = len(os.listdir(profile_path))
    print("Profile found with", image_count, "images.")
    images = [os.path.join(profile_path, x) for x in os.listdir(profile_path)]
    rep = np.zeros((len(images), 96 * 96))
    for i, im_path in enumerate(images):
        im = cv.imread(im_path, 0)
        cv.waitKey(1)
        rep[i, :] = preprocess_face(im)
    return rep


def preprocess_camera_image(img):
    """
    Preprocessing for face detection
    """
    grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
    working_image = cv.equalizeHist(working_image)
    working_image = cv.GaussianBlur(working_image, (5, 5), 0)
    return working_image, grey_image


def preprocess_face(grey_face):
    """
    Transforms a n x n image into a feature vector
    :param grey_face: ( n x n ) image in grayscale
    :return gray_face_vector:  ( 1 x EXTR_FRAME_SIZE^2) row vector with the preprocessed face
    """
    # Face preprocessing
    EXTR_FRAME_SIZE = 96
    """
     Apply preprocessing to balance the image (color/lightning), such    
      as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and 
      equalization (cv.equalizeHist).                                     
    """
    grey_face = cv.bilateralFilter(grey_face, 9, 75, 75)
    grey_face = cv.equalizeHist(grey_face)
    grey_face = cv.GaussianBlur(grey_face, (5, 5), 0)

    # resize the face
    grey_face = cv.resize(grey_face, (EXTR_FRAME_SIZE, EXTR_FRAME_SIZE))
    grey_face = grey_face.reshape(EXTR_FRAME_SIZE * EXTR_FRAME_SIZE).astype(np.float)
    grey_face -= grey_face.mean()
    grey_face /= grey_face.max()

    return grey_face


def online_face_recognition_adapted(profile_names,
                                    IncrementalKCenters,
                                    n_pictures=15,
                                    video_filename=None):
    
    """
    Run online face recognition with adaptive added labels.

    Parameters
    ----------
    profile_names : list
        List of user names used in create_user_profile()
    IncrementalKCenters : class
        Class implementing Incremental k-centers
    n_pictures : int
        Number of (labeled) pictures to use for each user_name
    video_filename : str
        .mp4 video file. If None, read from camera
    
    """
    
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [np.ones(p.shape[0]) * (i + 1)]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int)
    
    #  Generate model
    model = IncrementalKCenters(faces, labels, label_names)
    
    # Start camera
    if video_filename is None:
        cam = cv.VideoCapture(0)
    else:
        cam = cv.VideoCapture(video_filename)
        
    # Lists to store unkown detections 
    store_unknown = []
    store_unknown_idx = []
    
    while True:
        ret_val, img = cam.read()
        working_image, grey_image = preprocess_camera_image(img)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            
            # look for eye classifier
            local_image = img[y:(y + y_range), x:(x + x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]),
                             (0, 0, 255), 2)
                continue
            
            # select face
            local_image = grey_image[y:(y + y_range), x:(x + x_range)]
            x_t = preprocess_face(local_image)

            """
            Centroids are updated here
            """
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])

            """
            Hard HFS solution is computed here
            """
            label_scores = model.online_ssl_compute_solution()
            scores = [ll[1] for ll in label_scores]
            labels = [ll[0] for ll in label_scores]
            sorted_label_indices = np.argsort(scores)

            """
            Show results
            """
            for ii, ll_idx in enumerate(sorted_label_indices):
                label = labels[ll_idx]
                score = scores[ll_idx]
                
                # If the label is unknown
                if (label not in label_names) and (ll_idx<1):

                    # If the number of times the unknown faces seen is less than a constraint 
                    if len(store_unknown) <= n_pictures :
                        store_unknown.append(x_t)
                        store_unknown_idx.append(len(model.Y)-1)
                        color = (100, 100, 100)
                    
                    else :
                        store_unknown_idx.append(len(model.Y)-1)
                        store_unknown.append(x_t)
                        color = (100, 100, 100)
                        
                        # Compute distances between the face and centroids of unknown faces  
                        d = distance.cdist(np.array(store_unknown), np.array(store_unknown))[-1]
                        
                        # Normalize the resulted values 
                        d  = d/d.max()

                        # Compute the number of closest faces 
                        number = np.sum(d>0.6)
                        
                        # If the face has been seen enough times
                        if number > n_pictures : 
                            number_class = np.max(model.Y) + 1
                            label = 'NewPerson/NbrClass = ' + str(number_class)

                            # Add a new label to the list of labels 
                            label_names.append(label)

                            # Change their classes from "0" to the new class value 
                            idx = np.array(store_unknown_idx)[np.where(d>0.6)[0]]
                            model.Y[idx] = number_class

                            # Remove these faces from the list of "unknown" faces 
                            idx_ = np.where(d<=0.6)[0]     
                            store_unknown_idx = list(np.array(store_unknown_idx)[idx_])
                            store_unknown = list(np.array(store_unknown)[idx_])

                            # Choose a color for the frame
                            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][number_class % 3]                            
                    
                else:
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][ll_idx % 3]

                txt = label + "  " + ('%.4f' % score)
                cv.putText(img, txt, (p1[0], p1[1] - 5 - 10 * ii), cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                0.5 + 0.5 * (ii == len(scores) - 1), color)
                                                
            cv.rectangle(img, p1, p2, color, 2)

        cv.putText(img, "Face recognition: [s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]:
            break
        if key == ord('s'):
            # Save face
            print('saved')
            # cv.imwrite("frame.png", img)
            if not os.path.exists('results'):
                os.makedirs('results')
            image_name = os.path.join('results', 'frame.png')
            cv.imwrite(image_name, img)
            print("Image saved at", image_name)

            ## cv.waitKey(1)
    cv.destroyAllWindows()