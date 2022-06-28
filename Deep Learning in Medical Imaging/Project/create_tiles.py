import os
import cv2
import tifffile
import openslide
import numpy as np
from tqdm.notebook import tqdm

# -------------------------------------------------------------------------------------------------- #

def tile(img, mask, N=20, sz=128):
    '''
    This functions creates tiles from a whole slide image

    '''

    # Initializations
    result = []
    shape = img.shape

    # Padding img and mask to de devisible by tile size sz
    pad0, pad1 = (sz - shape[0]%sz), (sz - shape[1]%sz)
    img = np.pad(img,[[pad0//2,pad0-pad0//2], [pad1//2,pad1-pad1//2], [0,0]], constant_values=0)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2], [pad1//2,pad1-pad1//2], [0,0]], constant_values=0)
    
    # Change background from white to black to alleviate stains impact
    img[:,:,0][mask[:,:,0] == 0] = 0
    img[:,:,1][mask[:,:,0] == 0] = 0
    img[:,:,2][mask[:,:,0] == 0] = 0
    
    # Change color channel from RGB to HSV to pick more relevant tiles
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]

    # Reshape img and mask
    img = img.reshape(img.shape[0]//sz,sz, img.shape[1]//sz,sz,3)
    img1 = img1.reshape(img1.shape[0]//sz,sz, img1.shape[1]//sz,sz,1)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    img1 = img1.transpose(0,2,1,3,4).reshape(-1,sz,sz,1)
    
    # Select tiles with highest values in Hue channel
    temp = img1.reshape(img1.shape[0],-1).sum(-1)
    idxs = np.argsort(temp)[::-1][:N]
    img = img[idxs]

    # Save results
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
        
    return result

# -------------------------------------------------------------------------------------------------- #

def tile_no_mask(img, N=20, sz=128):
    '''
    This functions creates tiles from a whole slide image

    '''

    # Initializations
    result = []
    shape = img.shape

    # Padding img to be devisible by tile size sz
    pad0, pad1 = (sz - shape[0]%sz), (sz - shape[1]%sz)
    img = np.pad(img,[[pad0//2,pad0-pad0//2], [pad1//2,pad1-pad1//2], [0,0]], constant_values=255)

    # Reshape img 
    img = img.reshape(img.shape[0]//sz,sz, img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    
    # Select tiles with highest values in Hue channel
    temp = img.reshape(img.shape[0],-1).sum(-1)
    idxs = np.argsort(temp)[:N]
    img = img[idxs]

    # Save results
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
        
    return result

# -------------------------------------------------------------------------------------------------- #

def create_tiles(path_images, path_to_save, n, path_label_masks=None,unlabeled=False):
    '''
    This functions saves tiles for all train images

    '''    

    # Loop over images with masks
    if path_label_masks!=None:
        l = os.listdir(path_label_masks)
    else:
        l = os.listdir(path_images)[:300]
        
    for i,im in tqdm(enumerate(l)):
        
        if unlabeled:
            # Read TIF image
            img = tifffile.imread(path_images+im)
        
        else:
            # Read whole slide image
            wsi = openslide.OpenSlide(path_images+im)
            wsi_region = wsi.read_region((0,0), wsi.level_count-2, wsi.level_dimensions[-2])
            img = np.asarray(wsi_region)
            img = img[:,:,:3]
        
        if path_label_masks != None :

            # Read mask
            wsi_mask = openslide.OpenSlide(path_label_masks+im)
            wsi_region_mask = wsi_mask.read_region((0,0), wsi_mask.level_count-2, wsi_mask.level_dimensions[-2])
            mask = np.asarray(wsi_region_mask)
            mask = mask[:,:,:3]

            # Create tiles
            tiles = tile(img,mask)
        
            for j, t in enumerate(tiles):
                img,idx = t['img'],t['idx']
                if i == 0 and j == 0: 
                    # Fit Vahadane normalizer on first tile
                    n.fit(img)
                
                # Apply Vahadane normalizer on each tile
                img = n.transform(img)
                
                # Save tile as tif image
                tifffile.imwrite(path_to_save+im.split('.')[0]+'_'+str(idx)+'.tif', img)
        else:
            # Create tiles
            tiles = tile_no_mask(img)
            
            for j, t in enumerate(tiles):
                img,idx = t['img'],t['idx']
                
                # Apply Vahadane normalizer on each tile
                img = n.transform(img)
                
                # Save tile as tif image
                tifffile.imwrite(path_to_save+im.split('.')[0]+'_'+str(idx)+'.tif', img)
    
    return n

# -------------------------------------------------------------------------------------------------- #

def create_tiles_no_mask(l, path, path_to_save, n):
    '''
    This functions creates tiles from images without masks

    '''

    for im in l:
        # Read whole slide image
        wsi = openslide.OpenSlide(path+im+'.tiff')
        wsi_region = wsi.read_region((0,0), wsi.level_count-2, wsi.level_dimensions[-2])
        img = np.asarray(wsi_region)
        img = img[:,:,:3]

        # Extract tiles
        tiles = tile_no_mask(img)
        
        # Loop over tiles
        for j, t in enumerate(tiles):
            img,idx = t['img'],t['idx']
            
            # Apply Vahadane normalizer on each tile
            img = n.transform(img)

            # Save tile as tif image
            tifffile.imwrite(path_to_save+im.split('.')[0]+'_'+str(idx)+'.tif', img)




