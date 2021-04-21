import pydicom as dicom
import os
import cv2
import PIL

# make it True if you want in PNG format
# Specify the .dcm folder path
folder_path = "/VinBigData/train/"
# Specify the output jpg/png folder path
jpg_folder_path = "/VinBigData/train_jpg/"
images_path = os.listdir(folder_path)
cntr = 1
for n, image in enumerate(images_path):
    ds = dicom.dcmread(os.path.join(folder_path, image))
    pixel_array_numpy = ds.pixel_array
    pixel_array_numpy = (pixel_array_numpy / pixel_array_numpy.max()) * 255

    Y = (int(pixel_array_numpy.shape[0] / 10)) * 9
    X = int(pixel_array_numpy.shape[1] / 2)
    if pixel_array_numpy[Y, X] < pixel_array_numpy.mean():
        pixel_array_numpy = (pixel_array_numpy * -1) + 255

    if pixel_array_numpy.min() != 0:
        pixel_array_numpy = pixel_array_numpy - pixel_array_numpy.min()

    Dif = pixel_array_numpy.max() - pixel_array_numpy.min()
    # if Dif < 255:
    #     image.enhance(1 + (255 - Dif) / 255)
    # pixel_array_numpy = pixel_array_numpy * (255 / pixel_array_numpy.max())

    # print(pixel_array_numpy.max())
    print(str(cntr) + "- " + str(pixel_array_numpy.min()))
    image = image.replace('.dicom', '.jpg')
    cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
    cntr += 1
    if n % 50 == 0:
        print('{} image converted'.format(n))
