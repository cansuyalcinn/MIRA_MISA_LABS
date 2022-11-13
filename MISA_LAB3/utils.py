# It contains the functions that are used in the main file

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


# Preprocessing -- Skull removal step
def skull_removal(img, mask):
    """Skull removal step
    Args:
        img (SimpleITK image): input image
        mask (SimpleITK image): input mask
    Returns:
        SimpleITK image: skull removed image
    """
    brain_mask = mask>=1
    skull_strip = img*(brain_mask)
    return skull_strip

# Save the array as a new nifti image
def save_as_nifti(array, filename, reference_image):
    """Save array as nifti image

    Args:
        array (array): array to be saved
        filename (str): path to save
        reference_image (str): path of reference image
    """
    reference_image = sitk.ReadImage(reference_image)
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(image, filename)

# Read the list of patient ID's.
def read_patients_list(folder_path):
    """Read the list of patient ID's.
    folder_path (str): path to the folder containing the patient files
    """
    aux_patients_list = os.listdir(folder_path)
    list_patient = []
    for str1 in aux_patients_list:
        list_patient.append(str1.replace(".nii.gz", ""))
    return list_patient

def get_tissue_model_seg(cwd, atlas_name, method, list_patient, tm):
    """Get the tissue model segmentation
    Args:
        cwd (str): current working directory
        atlas_name (str): name of the atlas
        method (str): name of the method
        list_patient (list): list of patient ID's
        tm (str): tissue model
    Returns:
        list: list of tissue model segmentation
    """
    arr_seg_imgs =[]
    # read images for each patient one by one
    for p in tqdm(list_patient):
        im_path = str(cwd) + f'/data/testing-set/testing-images/{p}.nii.gz'
        mask_path = str(cwd) + f'/data/testing-set/testing-labels/{p}_3C.nii.gz'
        img_array1 = sitk.GetArrayFromImage(sitk.ReadImage(str(im_path)))
        label_array1 = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

        # Preprocessing - skull stripping
        img_array = skull_removal(img_array1, label_array1)

        # check the image maximum intensity
        max_value = int(np.max(img_array))
        max_value = max_value+1

        # get the maximum among the tissue
        lookup_table = np.argmax(tm, axis=1)+1 # the matching with intensity and tissue

        if max_value > len(lookup_table):
            # extend the lookup table accordingly.
            zeros = np.zeros(max_value-len(lookup_table)+1)
            lookup_table = np.append(lookup_table, np.array(zeros) ) # the filling for the exceptional intensities.

        array_int = np.array(img_array, dtype="uint16")
        new_img = lookup_table[array_int] #matching with the lookup table vs test images

        ## Apply skull stripping because increasing it by 1, increased the background intensity
        new_img = new_img*(label_array1>=1)

        # save the segmented image as a nifti image
        save_as_nifti(new_img.astype(np.float64), str(cwd) + f'/data/TM_segmentation/seg_{atlas_name}_{method}_{p}.nii', im_path)

        arr_seg_imgs.append(new_img)

    return arr_seg_imgs

def show_segmented_slice(img_list):
    """Show the segmented slice
    Args:
        img_list (list): list of segmented images
    """
    for im in img_list:
        plt.imshow(im[150], cmap = "gray")


def dice_score(image1, image2):
    """Calculate the dice score
    Args:
        image1 (SimpleITK image): input image array
        image2 (SimpleITK image): input image array
    Returns:
        float: dice score
    """
    image1[image1 > 0] = 1
    image2[image2 > 0] = 1
    intersection = np.sum(np.multiply(image1, image2))
    dice = 2 * intersection / (np.sum(image1) + np.sum(image2))
    return dice

# Calculate dice score for each patient and save as csv file
def get_dice_score(cwd, atlas_name, method, list_patient):
    ds_csf = []
    ds_gm = []
    ds_wm = []

    for i in range(len(list_patient)):
        seg_path = str(cwd) + f'/data/TM_segmentation/seg_{atlas_name}_{method}_{list_patient[i]}.nii'
        label_path = str(cwd) + f'/data/testing-set/testing-labels/{list_patient[i]}_3C.nii.gz'
        seg_image = sitk.ReadImage(seg_path)
        label_image = sitk.ReadImage(label_path)
        for i in range(1, 4):
            d = dice_score(seg_image == i, label_image == i)
            if i == 1:
                ds_csf.append(d)
            elif i == 2:
                ds_gm.append(d)
            else:
                ds_wm.append(d)

    # save the dice scores as a csv file
    df = pd.DataFrame({'CSF': ds_csf, 'GM': ds_gm, 'WM': ds_wm})
    df.to_csv(str(cwd) + f'/data/dice_scores_{atlas_name}_{method}.csv', index=False)

    return df

def image_registration(fixed_path, moving_path):
    """Give two images and the registration of both, and its transformation map will be given

    Args:
        fixed_path (str): fixed (template) image
        moving_path (str): moving image (image that will be transformed)

    Returns:
        sitk image, transformix map: transformed image and the transformation map
    """

    fixedImage = sitk.ReadImage(fixed_path)
    movingImage = sitk.ReadImage(moving_path)

    #Start registration settings
    elastixImageFilter = sitk.ElastixImageFilter() #Image filter object

    elastixImageFilter.LogToFileOn()
    elastixImageFilter.SetOutputDirectory("./")

    #Defining settings
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)

    #Run registration
    elastixImageFilter.Execute()

    #Get result image
    resultImage = elastixImageFilter.GetResultImage()

    #Transformation map
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    #Set resampler interpolator as NearestNeighbor
    transformParameterMap[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformParameterMap[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]    
    transformParameterMap[2]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    return resultImage, transformParameterMap

def transform_atlas(atlas_path_csf, atlas_path_gm, atlas_path_wm, transformParameterMap):
    """Transform the atlas images using the transformation map
    Args:
        atlas_path_csf (str): path of the csf atlas image
        atlas_path_gm (str): path of the gm atlas image
        atlas_path_wm (str): path of the wm atlas image
        transformParameterMap (transformix map): transformation map
    Returns:
        sitk image: transformed atlas images
    """
    atlas_csf = sitk.ReadImage(atlas_path_csf) #Read labels
    atlas_gm = sitk.ReadImage(atlas_path_gm) #Read labels
    atlas_wm = sitk.ReadImage(atlas_path_wm) #Read labels
    
    transformixImageFilter = sitk.TransformixImageFilter() #Create object transform matrix
    transformixImageFilter.SetTransformParameterMap(transformParameterMap) #Set with found transform map
    transformixImageFilter.SetMovingImage(atlas_csf) #set moving image
    transformixImageFilter.Execute() # Execute transform
    atlas_csf_registered = transformixImageFilter.GetResultImage() #Get result image

    transformixImageFilter.SetMovingImage(atlas_gm) #set moving image
    transformixImageFilter.Execute() # Execute transform
    atlas_gm_registered = transformixImageFilter.GetResultImage() #Get result image

    transformixImageFilter.SetMovingImage(atlas_wm) #set moving image
    transformixImageFilter.Execute() # Execute transform
    atlas_wm_registered = transformixImageFilter.GetResultImage() #Get result image

    return atlas_csf_registered, atlas_gm_registered, atlas_wm_registered   

def enlarge(lookup_table, max_value):
    if max_value > len(lookup_table):
        # extend the lookup table accordingly.
        zeros = np.zeros(max_value-len(lookup_table)+1)
        lookup_table = np.append(lookup_table, np.array(zeros) ) # the filling for the exceptional intensities.
    return lookup_table