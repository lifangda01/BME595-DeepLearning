from pylab import *
import gdal
import cv2
import os
import random
from googledrive import Drive

def get_RGB_overview(gdalObj, overviewNum):
    '''
        Get the RGB overview of a GDAL object (RGB image).
    '''
    nBands = gdalObj.RasterCount
    print "RasterCount...", nBands
    rBand = gdalObj.GetRasterBand (1)
    gBand = gdalObj.GetRasterBand (2)
    bBand = gdalObj.GetRasterBand (3)
    print "OverviewCount...", rBand.GetOverviewCount()
    rOverview = rBand.GetOverview(overviewNum)
    gOverview = gBand.GetOverview(overviewNum)
    bOverview = bBand.GetOverview(overviewNum)
    width = rOverview.XSize
    height = rOverview.YSize
    print "OverviewSize...", (width, height, nBands)
    overview = zeros((height, width, nBands), dtype=uint8)
    overview[:,:,0] = rOverview.ReadAsArray(0, 0, width, height) 
    overview[:,:,1] = gOverview.ReadAsArray(0, 0, width, height) 
    overview[:,:,2] = bOverview.ReadAsArray(0, 0, width, height) 
    return overview

def get_gray_overview(gdalObj, overviewNum):
    '''
        Get the gray overview of a GDAL object (gray scale image).
    '''
    nBands = 1
    gBand = gdalObj.GetRasterBand(1)
    print "RasterCount...", nBands
    print "OverviewCount...", gBand.GetOverviewCount()
    gOverview = gBand.GetOverview(overviewNum)
    width = gOverview.XSize
    height = gOverview.YSize
    print "OverviewSize...", (width, height, nBands)
    overview = zeros((height, width), dtype=uint8)
    overview[:,:] = gOverview.ReadAsArray(0, 0, width, height)         
    return overview

def overlay_mask_on_slide(slide, mask):
    '''
        Display the mask on the slice image.
    '''
    # Highlighting the slide region
    maskInv = cv2.bitwise_not(mask)
    highlight = zeros(slide.shape, dtype=uint8)
    for i in range(3):
        highlight[:,:,i] = cv2.bitwise_and(slide[:,:,i], maskInv)
    return highlight

def get_foreground_mask(slide):
    '''
        Get the foreground mask for a slide image.
    '''
    # HSV is more discriminating
    hsv = cv2.cvtColor(slide, cv2.COLOR_RGB2HSV)
    blurred = cv2.GaussianBlur(hsv, (5,5), 0)
    # S & !V gives the best result
    _, smask = cv2.threshold(blurred[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, vmask = cv2.threshold(blurred[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(smask, cv2.bitwise_not(vmask))
    # Erosion followed by dilation to remove background noise
    kernel = ones((15,15), uint8)
    mask = cv2.erode(mask, kernel)
    kernel = ones((7,7), uint8)
    mask = cv2.dilate(mask, kernel)
    return mask

def generate_patches_with_label(dataset_path, slideGdal, tmaskGdal, title, patchSize, numSkip, overviewNum=5):
    '''
        Generate patches that can be used for training the network.
    '''
    # Get the overviews for masking purposes
    slide = get_RGB_overview(slideGdal, overviewNum)
    sWidth = slide.shape[1]
    sHeight = slide.shape[0]
    tmask = get_gray_overview(tmaskGdal, overviewNum)
    tmWidth = tmask.shape[1]
    tmHeight = tmask.shape[0]
    print "Tumor mask size...", (tmWidth, tmHeight)
    # slide image is generally over-sized
    slide = slide[:tmHeight, :tmWidth]   
    # Get the foreground mask
    fmask = get_foreground_mask(slide)
    print "Foreground mask size...", (fmask.shape[1], fmask.shape[0])
    tempmask = zeros((tmHeight, tmWidth)).astype(uint8)
    width = tmWidth * slideGdal.RasterXSize / sWidth
    height = tmHeight * slideGdal.RasterYSize / sHeight 
    print "Slide size...", (width, height)
    # Size ratio of original slide over mask
    ratio = float(width) / tmWidth
    count = 0
    print "Total number of possible patches...", width*height / patchSize**2
    # Iterate through all the patches
    for x in range(0, width-patchSize, patchSize):
        # print "Processing patch number...", count
        for y in range(0, height-patchSize, patchSize):
            count += 1
            # Get the corresponding pixel coordinate in masks
            my = int(round(y / ratio))
            mx = int(round(x / ratio))
            try:
                # Is this foreground?
                if fmask[my,mx] == 0: 
                    continue
                # Does this contain cancer?
                # Function usage: gdalObj.ReadAsArray(x,y,xoff,yoff)
                if tmaskGdal.ReadAsArray(x+patchSize/2, y+patchSize/2, 1, 1)[0,0] == 0:
                    label = 0
                else:
                    label = 1
                # Skip some normal images since there are too many of them
                if label == 0 and random.randint(0,numSkip) != 0: continue
                # Get the patch from the GDAL object
                patch = slideGdal.ReadAsArray(x, y, patchSize, patchSize)
                patch = swapaxes(patch, 0, 2)
                if label == 1:
                    imsave(dataset_path + "/train/tumor/" + "_".join([title, str(count), str(label)]) + ".png", patch)
                else:
                    imsave(dataset_path + "/train/normal/" + "_".join([title, str(count), str(label)]) + ".png", patch)
            except IndexError:
                pass

def generate_training_dataset(dataset_path, patchSize=224, numSkip=10):
    '''
        Download and preprocess tumor image with mask one by one.
    '''
    # Establish connection with Google Drive
    drive = Drive()
    tumor_file_list = drive.get_file_list(drive.train_tumor_id)
    mask_file_list = drive.get_file_list(drive.ground_truth_mask_id)
    # Preprocess one tumor image at a time
    for tumor_file in tumor_file_list:
        # Download tumor file
        tumor_file_title = tumor_file['title']
        # Download mask file
        num = tumor_file_title.split('_')[1].split('.')[0]
        mask_file_title = "Tumor_{0}_Mask.tif".format(num)
        for mf in mask_file_list: 
            if mf['title'] == mask_file_title: 
                mask_file = mf
        print "Preprocessing...", tumor_file_title, mask_file_title
        print "Downloading began..."
        tumor_file_path = os.path.join('temp', tumor_file_title)
        mask_file_path = os.path.join('temp', mask_file_title)
        try:
            tumor_file.GetContentFile(tumor_file_path)
            mask_file.GetContentFile(mask_file_path)
        except:
            continue
        print "Downloading finished..."
        print "Generating patches..."
        slideGdal = gdal.Open(tumor_file_path)
        tmaskGdal = gdal.Open(mask_file_path)
        generate_patches_with_label(dataset_path, slideGdal, tmaskGdal, num, patchSize, numSkip)
        print "Patch generation finished..."
        os.remove(tumor_file_path)
        os.remove(mask_file_path)

def dataset_split(dataset_path, split_path, ratio):
    '''
        Split the generated dataset by ratio into split path.
    '''
    tumor_file_list = array([f for f in os.listdir(os.path.join(dataset_path, 'tumor')) if f.endswith('.png')])
    normal_file_list = array([f for f in os.listdir(os.path.join(dataset_path, 'normal')) if f.endswith('.png')])
    num_total_tumor = len(tumor_file_list)
    num_total_normal = len(normal_file_list)
    num_test_tumor = int(num_total_tumor * (1-ratio))
    num_test_normal = int(num_total_normal * (1-ratio))
    print "num_total_tumor, num_total_normal = ", num_total_tumor, num_total_normal
    print "num_test_tumor, num_test_normal = ", num_test_tumor, num_test_normal
    tumor_indices = arange(num_total_tumor)
    random.shuffle(tumor_indices)
    tumor_test_indices = tumor_indices[:num_test_tumor]
    normal_indices = arange(num_total_normal)
    random.shuffle(normal_indices)
    normal_test_indices = normal_indices[:num_test_normal]
    for f in tumor_file_list[tumor_test_indices]:
        os.rename(os.path.join(dataset_path, 'tumor',f), os.path.join(split_path, 'tumor',f))
    for f in normal_file_list[normal_test_indices]:
        os.rename(os.path.join(dataset_path, 'normal',f), os.path.join(split_path, 'normal',f))


# Use gdalinfo for readable information about image    
# imshow displays RGB image (not BGR)    
def main():
    generate_training_dataset('./images1', patchSize=224, numSkip=10)

if __name__ == "__main__":
    main()
