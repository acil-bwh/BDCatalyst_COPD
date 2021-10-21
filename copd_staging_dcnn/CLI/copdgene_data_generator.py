# COPDGene training data generator
import os
import random
import glob

import numpy      as np
import SimpleITK  as sitk
import tensorflow as tf


def batchGenerator(file_list, label_list, batch_size):
    # Load first NRRD
    file_index = 0
    slice_num  = 0
    img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)

    # Loop indefinitely
    while True:
        # Initialize image batch
        batch_array = []
        labels      = []

        # Populate array until we hit the batch size
        while len(batch_array) < batch_size:
            # if slice_num < img_array.shape[-1] - 1:  # img_array.shape[0]-1:
            #     batch_array.append(img_array[slice_num, :, :])
            #     labels.append(img_label)
            #     slice_num += 1
            # else:
            #     img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)
            #     slice_num = 0
            batch_array.append(img_array)
            labels.append(img_label)
            img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)
            slice_num = 0

        # Set correct formats
        batch_array = np.array(batch_array)
        # print(f'batch_array shape BEFORE: {batch_array.shape}')
        batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3], 1))
        # print(f'batch_array shape AFTER: {batch_array.shape}')
        labels = np.array(labels)
        # labels = tf.keras.utils.to_categorical(labels)

        # Yield to the calling function
        yield (batch_array, labels)


def batchGeneratorPredict(file_list, label_list, batch_size, small_set=32):
    # Load first NRRD
    file_index = 0
    slice_num  = 0
    slice_all  = 0
    counter    = 0

    img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)

    # img_array = img_array[:, :, :small_set]

    # Loop indefinitely
    while slice_all < img_array.shape[-1] - 1:
        # Initialize image batch
        batch_array = []
        labels = []

        # Populate array until we hit the batch size
        while len(batch_array) < min(batch_size, small_set):
            # if slice_num < img_array.shape[-1] - 1:  # img_array.shape[0]-1:
            #     batch_array.append(img_array[:, :, slice_num])
            #     labels.append(img_label)
            #     slice_num += 1
            # else:
            #     if len(file_list) > 1:
            #         img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)
            #     slice_all += slice_num
            #     counter   += 1
            #     slice_num  = 0
            if len(file_list) > 1:
                batch_array.append(img_array)
                labels.append(img_label)
                img_array, img_label, file_index = processNextNrrd(file_list, label_list, file_index)
            slice_all += slice_num
            counter += 1
            slice_num = 0

        # Set correct formats
        batch_array = np.array(batch_array)

        print(f'batch_array shape BEFORE: {batch_array.shape} Counter: {counter}')
        batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3], 1))

        print(f'batch_array shape AFTER: {batch_array.shape}')
        labels = np.array(labels)

        # Yield to the calling function
        yield (batch_array, labels)


def processNextNrrd(file_list, label_list, file_index):
    # Load nrrd file
    img = sitk.ReadImage(file_list[file_index])
    # Convert to img_array
    img_array = sitk.GetArrayFromImage(img)
    # Clip values [-1024, 50]
    img_array[img_array<-1024] = -1024
    img_array[img_array>   50] =    50
    # Normalize 0-1
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    # Pull label
    img_label = label_list[file_index]

    # If not at the end of the file list, queue the next nrrd; otherwise loop to the beginning
    if file_index < len(file_list) - 1:
        file_index += 1
    else:
        file_index = 0

    return img_array, img_label, file_index


def pullRandomNrrds(parent_dir, insp_exp='', std_sharp='', num_files=100):
    file_list   = []
    file_labels = []

    subject_list = glob.glob(os.path.join(parent_dir, '*/'))
    ##### TODO: Replace the subject labels with something useful from PIC-SURE
    ##### For now just assign them a random int from 0 to 5 for our classifier
    subject_label_list = [random.randint(0, 5) for subject in subject_list]

    while (len(file_list) < num_files) and (len(subject_list) > 0):
        file_index    = random.randrange(len(subject_list))
        subject       = subject_list.pop(file_index)
        subject_label = subject_label_list.pop(file_index)
        file_name     = glob.glob(os.path.join(subject, '*' + insp_exp + std_sharp + '.nrrd'))  # EDITED
        labels        = [subject_label for file in file_name]
        if not file_name:
            continue
        else:
            for file in file_name:
                file_list.append(file)
            for label in labels:
                file_labels.append(label)

    print(f'Returned {len(file_list)} files')
    return file_list, file_labels


def getImageSetSize(file_list, index_first=True):
    file_size_list = []
    reader = sitk.ImageFileReader()


    index = 0 if index_first else -1

    for file in file_list:
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        # file_size_list.append(reader.GetSize()[index])  # (554, 512, 512)
        file_size_list.append(1)

    num_images = sum(file_size_list)

    return file_size_list, num_images