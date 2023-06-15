# This file gives an example of how to run the onnx model with any data
# The data can either be a DICOM directory or a nifti file (seen in main)
# The code in the run_models function is inspired by code in the inference
# section of the nnUnet codebase that is required to confirm the proper usage of the onnx model
#
# main() can easily be adjusted to loop through a dataset - SET the filenames and the path to the models
# Author : Alice Santilli <santila@mskcc.org>

from batchgenerators.augmentations.utils import pad_nd_image
import gc
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import SimpleITK as sitk
import torchio as tio



from segment_utils import (
    calculate_gaussian_filter,
    compute_steps_for_sliding_window,
    internal_3Dconv_tiled,
    left_squeeze,
)

PATCH_SIZE_LIST = [224,96,112]
STEP_SIZE = 0.5
USE_GAUSSIAN = True
NUM_CLASSES = 2
ADD_ACCT = True

def main():
    ### Using dicom folders
    filename_pet = "33857242/S0012_PET-AC, AX, Ga68 DOTATATE"
    filename_ac = "33857242/S0002_CT-AC, Ga68 DOTATATE, Trans"
    pet_img = load_full_itk(filename_pet)
    ac_img = load_full_itk(filename_ac)

    ### Using nifti images
    #filename_pet = '/Users/asantilli/Desktop/cars_pres_case/35561800/33857242/S0012_PET-AC, AX, Ga68 DOTATATE.nii.gz'
    #filename_ac = '/Users/asantilli/Desktop/cars_pres_case/35561800/33857242/S0002_CT-AC, Ga68 DOTATATE, Trans.nii.gz'
    #pet_img = load_nifti(filename_pet)
    #ac_img = load_nifti(filename_ac)

    #Preproce images
    normalized_pet, normalized_ac = preprocess_images(pet_img, ac_img)

    if ADD_ACCT:
        # stack the two images as 2 channels
        model_image = np.squeeze(np.asarray([normalized_pet, normalized_ac]))
        print(model_image.shape)
        model_name = "Task511_PETAC_5fold_nnunet.onnx"
        seg_save = '33857242/pet_ac_onnx_prediction.nii.gz'
    else :
        model_image = normalized_pet
        model_name = "Task521_PET_5fold_nnunet.onnx"
        seg_save = '33857242/pet_onnx_prediction.nii.gz'


    #Run Model
    predicted_seg = run_onnx_model(model_image, model_name)
    print(predicted_seg.shape)
    if len(predicted_seg.shape) != 4:
        predicted_seg = np.expand_dims(predicted_seg,0)
    final_seg = tio.ScalarImage(tensor=predicted_seg, affine=pet_img.affine)
    final_seg.save(seg_save)



def preprocess_images(pet_img, ac_img):
    # PREPROCESS PET
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))

    pet_pp_img = rescale(pet_img)

    # PREPROCESS AC/CT

    dim = ac_img.shape[3]
    transform = tio.CropOrPad((128, 128, dim))
    clamp = tio.Clamp(out_min=(50 - 160), out_max=(50 + 160))

    ac_crop = transform(ac_img)
    ac_pp_img = rescale(clamp(ac_crop))

    # onnx model is flipped (c, z, x, y )
    pet_model_image = np.transpose(pet_pp_img, axes=[0, 3, 2, 1])
    ac_model_image = np.transpose(ac_pp_img, axes=[0, 3, 2, 1])

    print("Preprocessed Images")
    print("PET ", pet_model_image.shape)
    print("ACCT ", ac_model_image.shape)

    #plt.imshow(pet_model_image[0, 180, :, :], cmap='binary')
    #plt.show()
    #plt.imshow(ac_model_image[0, 180, :, :], cmap='binary')
    #plt.show()
    return pet_model_image, ac_model_image

def load_full_itk(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    print("Image size :" ,size)
    img = tio.ScalarImage.from_sitk(image)
    return img

def load_nifti(filename):
    img = tio.ScalarImage(filename)
    print("Image size: ", img.shape)
    return img


def run_onnx_model(image, model_name):

    patch_size_list = [int(x) for x in PATCH_SIZE_LIST]
    patch_size = tuple(patch_size_list)

    # compute steps for sliding window
    steps, num_tiles = compute_steps_for_sliding_window(
        patch_size, image.shape[1:], step_size=STEP_SIZE
    )
    if USE_GAUSSIAN:
        gaussian_importance_map, add_for_nb_of_preds = calculate_gaussian_filter(
            patch_size=patch_size, num_tiles=num_tiles
        )
        print("Gaussian map shape", gaussian_importance_map.shape)
    aggregated_results = np.zeros(
        [NUM_CLASSES] + list(image.shape[1:]), dtype=np.float32
    )
    aggregated_nb_of_predictions = np.zeros(
        [NUM_CLASSES] + list(image.shape[1:]), dtype=np.float32
    )
    # for sliding window inference the image must at least be as large as the patch size. It does not matter
    # whether the shape is divisible by 2**num_pool as long as the patch size is
    pad_image, slicer = pad_nd_image(
        image,
        patch_size,
        mode="constant",
        kwargs={"constant_values": 0},
        return_slicer=True,
    )

    batches = internal_3Dconv_tiled(
        pad_image,
        patch_size,
        steps,
        STEP_SIZE,
        NUM_CLASSES,
        do_mirroring=False,
        mirror_axes=(0, 1, 2),
    )

    # create ONNX session
    ort_session = ort.InferenceSession(model_name)

    for data, lb_x, ub_x, lb_y, ub_y, lb_z, ub_z in batches:
        batch = data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]
        batch_blank = np.concatenate((batch, batch), axis=0).astype(np.float32)
        print(f"Batch shape: {batch_blank.shape}")
        inputs = ort_session.get_inputs()[0]
        input_name = inputs.name
        ort_inputs = {input_name: batch_blank}
        predicted_patch = ort_session.run(None, ort_inputs)[0][0]
        # apply gaussian
        predicted_patch[:] *= gaussian_importance_map
        aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
        aggregated_nb_of_predictions[
        :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z
        ] += add_for_nb_of_preds
    del predicted_patch
    print("aggregated_results", aggregated_results.shape)
    print("aggregated max", np.amax(aggregated_results))
    gc.collect()
    # we reverse the padding here (remember that we padded the input to be at least as large as the patch size
    slicer = tuple(
        [
            slice(0, aggregated_results.shape[i])
            for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
        ]
        + slicer[1:]
    )
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

    # computing the class_probabilities by dividing the aggregated result with result_numsamples
    aggregated_results /= aggregated_nb_of_predictions
    del aggregated_nb_of_predictions
    predicted_segmentation = np.transpose(aggregated_results.argmax(0), axes=[2, 1, 0])
    return predicted_segmentation



if __name__ == '__main__':
    main()