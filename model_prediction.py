# detect VinBigData in photos with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset

import os


# os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"


# class that defines and loads the VinBigData dataset
class VinBigDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class

        self.add_class("dataset", 1, "Aortic enlargement")
        self.add_class("dataset", 2, "Atelectasis")
        self.add_class("dataset", 3, "Calcification")
        self.add_class("dataset", 4, "Cardiomegaly")
        self.add_class("dataset", 5, "Consolidation")
        self.add_class("dataset", 6, "ILD")
        self.add_class("dataset", 7, "Infiltration")
        self.add_class("dataset", 8, "Lung Opacity")
        self.add_class("dataset", 9, "Nodule/Mass")
        self.add_class("dataset", 10, "Other lesion")
        self.add_class("dataset", 11, "Pleural effusion")
        self.add_class("dataset", 12, "Pleural thickening")
        self.add_class("dataset", 13, "Pneumothorax")
        self.add_class("dataset", 14, "Pulmonary fibrosis")
        self.add_class("dataset", 15, "No finding")
        # define data locations
        images_dir = dataset_dir + '/train_jpg/'
        annotations_dir = dataset_dir + '/train_xml/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        id_lookup = dict()
        id_lookup['Aortic enlargement'] = 1
        id_lookup['Atelectasis'] = 2
        id_lookup['Calcification'] = 3
        id_lookup['Cardiomegaly'] = 4
        id_lookup['Consolidation'] = 5
        id_lookup['ILD'] = 6
        id_lookup['Infiltration'] = 7
        id_lookup['Lung Opacity'] = 8
        id_lookup['Nodule/Mass'] = 9
        id_lookup['Other lesion'] = 10
        id_lookup['Pleural effusion'] = 11
        id_lookup['Pleural thickening'] = 12
        id_lookup['Pneumothorax'] = 13
        id_lookup['Pulmonary fibrosis'] = 14
        id_lookup['No finding'] = 15
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        class_ids = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        for obj in root.findall('.//object'):
            class_ids.append(int(id_lookup[str(obj.find('name').text)]))

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height, class_ids

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h, c_ids = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        return masks, asarray(c_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "VinBigData_cfg"
    # number of classes (background + classes)
    NUM_CLASSES = 1 + 15
    # number of training steps per epoch
    STEPS_PER_EPOCH = 150
    # Min and max image resize and padding
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 3072
    # channels
    IMAGE_CHANNEL_COUNT = 3
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


import random


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=3):
    # load image and mask
    for i in range(n_images):
        # load the image and mask 0005e8e3701dfb1dd93d53e2ff537b6e
        image = dataset.load_image(i + 5)
        mask, _ = dataset.load_mask(i + 5)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        print(len(sample), model.config.BATCH_SIZE)
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual ' + str(image[(0,) * image.ndim]))
        # pyplot.savefig('actul.png')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i * 2 + 2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted ' + str(image[(0,) * image.ndim]))
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.savefig(f'VinBigDataTest{random.randint(0, 103543)}.png')
    pyplot.show()


# load the train dataset
train_set = VinBigDataset()
train_set.load_dataset('VinBigData', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = VinBigDataset()
test_set.load_dataset('VinBigData', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'vinbigdata_cfg20210419T1623/mask_rcnn_vinbigdata_cfg_0005.h5'
model.load_weights(model_path, by_name=True)

# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, cfg)
