import os 
import PIL
import cv2
import torch
import tifffile
import openslide
import matplotlib
import numpy as np
from torch import nn
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set()

# -------------------------------------------------------------------------------------------------- #

def wsi_info(train_df, path_train_images, image):
    '''
    This function displays information about a given Whole Slide Image (WSI)

    '''

    # Read a WSI 
    wsi = openslide.OpenSlide(path_train_images+image+'.tiff')
    wsi_region = wsi.read_region((0,0), wsi.level_count-1, wsi.level_dimensions[-1])
    
    # Compute physical size of a pixel converted to microns
    spacing = 1 / (float(wsi.properties['tiff.XResolution']) / 10000) 

    # Display informations : ID / Dimensions / Spacing / Downsapling factors / Gleason Score / ISUP Grade 
    print(f"File id: {image}")
    print(f"Dimensions: {wsi.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Downsample factor per level: {wsi.level_downsamples}")
    print(f"Dimensions of levels: {wsi.level_dimensions}\n")
    print(f"Gleason score: {train_df.loc[image, 'gleason_score']}")
    print(f"ISUP grade: {train_df.loc[image, 'isup_grade']}")

    # Display image
    _ , ax =  plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(wsi_region) 
    ax[0].set_title('Full Image')
    ax[1].imshow(wsi.get_thumbnail(size=(256,256))) 
    ax[1].set_title('Zoomed Image')
    ax[0].grid(False)
    ax[1].grid(False)
    plt.show()

# -------------------------------------------------------------------------------------------------- #

def show_masks(train_df, path_label_masks, list_mask):
    '''
    This function shows masks of a given list of ids

    '''

    fig, axs = plt.subplots(2,2,figsize=(15,20))

    # Loop over the samples Ids
    for i, id in enumerate(list_mask):
        
        # Collect information for each sample
        data_provider = train_df.loc[id, 'data_provider']
        isup_grade = train_df.loc[id, 'isup_grade']
        gleason_score = train_df.loc[id, 'gleason_score']

        # Read the WSI mask
        wsi_mask = openslide.OpenSlide(path_label_masks+id+'.tiff')
        wsi_mask_region = wsi_mask.read_region((0,0), wsi_mask.level_count - 1, wsi_mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red']) # for labels [0,1,2,3,4,5]
        pos = axs[(i-1)//2,(i-1)%2].imshow(np.asarray(wsi_mask_region)[:,:,0], cmap=cmap,vmin=0, vmax=5)
        wsi_mask_region.close()    

        axs[(i-1)//2,(i-1)%2].axis('off')
        axs[(i-1)//2,(i-1)%2].set_title(f"ISUP: {isup_grade} \nGleason: {gleason_score}\nSource: {data_provider}\nID: {id}")

    # Show the color bar of ISUP grades
    cbar = fig.colorbar(pos,ax=axs,shrink=0.4, pad=0.05,orientation = 'horizontal')
    cbar.ax.set_yticklabels(['0: background','1: stroma','2: healthy epithelium','3: cancerous epithelium\n    gleason 3','4: cancerous epithelium \n    gleason 4','5: cancerous epithelium\n    gleason 5']);
    plt.show()

# -------------------------------------------------------------------------------------------------- #

def overlay_mask_on_slide(train_df, path_train_images, path_label_masks, images, alpha=0.8, max_size=(256, 256)):
    '''
    This function shows masks overlayed on corresponding images

    '''

    _ , ax = plt.subplots(1,3, figsize=(18,22))
    
    # Loop over the given images
    for i, image_id in enumerate(images):
        
        # Read the WSI sample and its mask
        slide = openslide.OpenSlide(path_train_images+image_id)
        mask = openslide.OpenSlide(path_label_masks+image_id)
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]

        # Create alpha mask
        alpha_int = int(round(255*alpha))

        id = image_id.split('.')[0]
        center = train_df.loc[id, 'data_provider']

        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)
        alpha_content = PIL.Image.fromarray(alpha_content)
        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        # Overlay mask on image
        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        # Visualization
        ax[i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i%3].axis('off')
        
        # Show information
        id = image_id.split('.')[0]
        data_provider = train_df.loc[id, 'data_provider']
        isup_grade = train_df.loc[id, 'isup_grade']
        gleason_score = train_df.loc[id, 'gleason_score']
        ax[i%3].set_title(f"ID: {id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

# -------------------------------------------------------------------------------------------------- #

def before_after_mask(path_train_images, working_dir, name='4a3e6e77b42a7e8f8b2dec472667bdbe'):
    '''
    This function compares tiles before and after applying mask

    '''

    # Read stained whole slide image
    stained_wsi_path = path_train_images + name + '.tiff'
    stained_wsi = openslide.OpenSlide(stained_wsi_path)
    stained_wsi = stained_wsi.read_region((0,0), stained_wsi.level_count-1, stained_wsi.level_dimensions[-1])

    # Display stained whole slide image
    plt.imshow(np.asarray(stained_wsi))
    plt.title('Original Image')
    plt.show()

    # Read tiles before and after
    path_1 = working_dir + 'train_tiles_20/'
    path_2 =  working_dir + 'train_clean_normalized/'
    stained_tiles_before = [cv2.imread(path_1+name+'_'+str(i)+'.jpg') for i in range(20)]
    stained_tiles_after = [tifffile.imread(path_2+name+'_'+str(i)+'.tif') for i in range(20)]

    # Display tiles before
    _ , axs = plt.subplots(1,20, figsize=(30,3))
    for i in range(20):
        axs[i].imshow(stained_tiles_before[i])
        axs[i].grid(False)
    plt.suptitle('Before Applying Masks')
    plt.show()

    print()

    # Display tiles after
    _ , axs = plt.subplots(1,20, figsize=(30,3))
    for i in range(20):
        axs[i].imshow(stained_tiles_after[i])
        axs[i].grid(False)
        plt.suptitle('After Applying Masks')
    plt.show()

# -------------------------------------------------------------------------------------------------- #

class Loss(): 

  def ordinal_regression(self, predictions,targets):
      '''
      Reference: https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99

      '''

      # Create out modified target with [batch_size, num_labels] shape
      modified_target = torch.zeros_like(predictions)

      # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
      for i, target in enumerate(targets):
          modified_target[i, 0:target+1] = 1
      
      return nn.MSELoss(reduction='mean')(predictions, modified_target)

# -------------------------------------------------------------------------------------------------- #

def read_data(working_dir, train_df, split=True):
    """
    This function reads concatenated images for the ResNet18 and pooling_restnet18 models 
    and apply if "split = True" the train test split method

    """

    # Read the list of images Ids
    path_train_images = working_dir + 'train_clean_n_concatenated'
    l = os.listdir(path_train_images)

    X = []
    y = []

    # Read TIF images with their corresponding ISUP gardes
    for im in tqdm(l):
        img = tifffile.imread(path_train_images+'/'+im)
        X.append(img)
        y.append(train_df.loc[im[:-4]]['isup_grade'])
    X = np.array(X)
    y = np.array(y)

    # Reshape the data for the modeling phase 
    X1 = np.zeros((340, 3, 640, 512))
    for i in range(X.shape[0]):
        X_ = X[i]
        X1[i,0,:,:] = X_[:,:,0]
        X1[i,1,:,:] = X_[:,:,1]
        X1[i,2,:,:] = X_[:,:,2]
    
    # Normalize the data
    X=X1 / 255

    # Train test split 
    if split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 0)
        return X_train, X_val, y_train, y_val
    
    return X, y 

# -------------------------------------------------------------------------------------------------- #

def read_data_augmented(working_dir, train_df): 
  """
  This function reads the concatenated images for the ResNet18 and pooling_restnet18 models 
  and applies data augmentation tarnsformations
  
  """

  # Read the list of images Ids
  path_train_images = working_dir + 'train_clean_n_concatenated'
  l = os.listdir(path_train_images)

  # Data transformations
  data_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        ])

  X = []
  y = []

  # Read TIF images with their corresponding ISUP gardes
  for im in l:
    img = tifffile.imread(path_train_images+'/'+im)
    img = Image.fromarray(img)

    # Apply data transformations 
    img = data_transform(img).numpy()
    
    X.append(img)
    y.append(train_df.loc[im.split('.')[0]]['isup_grade'])
  
  X = np.array(X)
  y = np.array(y)

  return X , y 

# -------------------------------------------------------------------------------------------------- #

def show(train_auc, val_auc, train_acc, val_acc):
    """
    This function displays the evolution of the training and validation 
    AUC and Accuracy curves in function of number of epochs  

    """
    _ , ax = plt.subplots(1,2 , figsize=(15,5))
    if len(val_auc)!= 0:
        ax[0].plot(np.arange(len(train_auc)), train_auc, label='Training')
        ax[0].plot(np.arange(len(val_auc)),val_auc,label='Validation')
        ax[0].legend()
        ax[0].set_title('AUC')
        ax[0].set_xlabel('epochs')

        ax[1].plot(np.arange(len(train_acc)), train_acc, label='Training')
        ax[1].plot(np.arange(len(val_acc)),val_acc,label='Validation')
        ax[1].legend()
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('epochs')
    else: 
        ax[0].plot(np.arange(len(train_auc)), train_auc, label='Training')
        ax[0].legend()
        ax[0].set_title('AUC')
        ax[0].set_xlabel('epochs')

        ax[1].plot(np.arange(len(train_acc)), train_acc, label='Training')
        ax[1].legend()
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('epochs')

    plt.show()

# -------------------------------------------------------------------------------------------------- #

def read_unlabeled_data(working_dir, size=300):
    """
    This function reads unlabled data for the pooling_restnet18 model for the semi-supervised task

    """

    # Read the list of images Ids
    path_train_images = working_dir + 'Unlabeled_data/all_tiles_n_concatenated'
    l = os.listdir(path_train_images)[:size]

    X_unlabled = []

    # Read TIF images 
    for im in tqdm(l):
        img = tifffile.imread(path_train_images+'/'+im)
        X_unlabled.append(img)
    X_unlabled = np.array(X_unlabled)

    # Reshape the data for the modeling phase 
    X1 = np.zeros((size, 3, 640, 512))
    for i in range(X_unlabled.shape[0]) :
        X_ = X_unlabled[i]
        X1[i,0,:,:] = X_[:,:,0]
        X1[i,1,:,:] = X_[:,:,1]
        X1[i,2,:,:] = X_[:,:,2]
    
    return X1/255

# -------------------------------------------------------------------------------------------------- #

def read_test_data(path_test_images):
    """
    This function reads test data for the ResNet18 and pooling_restnet18 models 
    
    """

    # Read the list of images Ids
    l = os.listdir(path_test_images)

    X_test = []
    test_Id = []

    # Read TIF images 
    for im in tqdm(l):
        img = tifffile.imread(path_test_images+'/'+im)
        X_test.append(img)
        test_Id.append(im.split('.')[0])
    X_test = np.array(X_test)

    # Reshape the data for the modeling phase 
    X1 = np.zeros((86, 3, 640, 512))
    for i in range(X_test.shape[0]) :
        X_ = X_test[i]
        X1[i,0,:,:] = X_[:,:,0]
        X1[i,1,:,:] = X_[:,:,1]
        X1[i,2,:,:] = X_[:,:,2]
    
    return X1/ 255, test_Id

# -------------------------------------------------------------------------------------------------- #

def read_test_data_MIL(path_test_images):
    """
    This function reads test data for MIL model
    
    """

    # Read the list of images Ids
    l = os.listdir(path_test_images)

    test_Id = []
    X_test = []
    X_20 = []

    # Read TIF images 
    for i,im in enumerate(tqdm(l)):
        img = tifffile.imread(path_test_images+im)
        
        # Reshape the image for the modeling phase
        X1 = np.zeros((3,128,128))
        X1[0,:,:] = img[:,:,0]
        X1[1,:,:] = img[:,:,1]
        X1[2,:,:] = img[:,:,2]
        img = X1
        
        X_20.append(img)

        if (i+1) % 20 == 0 :
            X_test.append(X_20)
            test_Id.append(im.split('_')[0])
            X_20=[]

    return np.array(X_test)/255, test_Id

# -------------------------------------------------------------------------------------------------- #

def read_data_MIL(path_images, train_df, augmented=True, labeled=True):
    """"
    This function reads data for the MIL model in the augmented/non-augmented and the labeled/unlabeled cases 

    """
    # Data transformations 
    data_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        ])

    # Read the list of images Ids
    if labeled : 
      l = train_df.index
    
    else:
      l = os.listdir(path_images)
      l1 = []
      for x in l :
        l1.append(x.split('_')[0])
      l = list(set(l1))
      
    data = []
    label = []
    X_20 = []

    # Read TIF images 
    for im in tqdm(l):

        idx = np.arange(20)
        for i in idx : 
          img = tifffile.imread(path_images+"/"+im+"_"+str(i)+".tif")
          img = Image.fromarray(img)
          
          # Apply data transformations 
          if augmented:
              img = data_transform(img).numpy()*255

          else : 
            # Reshape the image for the modeling phase 
            img = np.array(img)
            X1 = np.zeros((3,128,128))
            X1[0,:,:] = img[:,:,0]
            X1[1,:,:] = img[:,:,1]
            X1[2,:,:] = img[:,:,2]
            img = X1
        
          X_20.append(img)
          
        data.append(X_20)
        X_20=[]
        
        # Read the corresponding ISUP grades
        if labeled:
            label.append(train_df.loc[im.split('_')[0]]['isup_grade'])
 
    # Normalize the data
    data = np.array(data)/255
    
    if labeled:
        label = np.array(label)
        return data , label 

    return data


