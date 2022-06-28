import os
import tifffile
import numpy as np
from tqdm.notebook import tqdm

# -------------------------------------------------------------------------------------------------- #

def concatenate_tiles(path_tiles, path_to_save, path=None,no_mask=False):
    '''
    This function is created to concatenate extracted tiles 
    
    '''

    # Read the tiles Ids
    if no_mask:
        p = path_tiles
    else:
        p = os.listdir(path_tiles)

    names = [f.split('_')[0] for f in p]
    names = list(set(names))

    for x in tqdm(names):
        img = []
        idx = np.arange(0,20)
        
        # Read TIF tile image
        for k, i in enumerate(idx): 
            img_name = x+ '_'+str(i)+'.tif'
            if path!=None:
                a=tifffile.imread(path+img_name)
            else:
                a=tifffile.imread(path_tiles+img_name)
            
            # Concatenate tiles
            img.extend(a)
            if k == 4 :
                new_image = np.array(img)
                img = []
            if k in [9, 14, 19, 24] :
                new_image = np.hstack((new_image, np.array(img)))
                img = []
        
        # Save concatenated image
        tifffile.imwrite(path_to_save+x+'.tif', new_image)