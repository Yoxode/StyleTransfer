CONTENT_PATH = "contents/content1.jpg"
STYLE_PATH = "styles/style1.jpg"
OUTPUT_PATH = "outputs/output"
VGG_URL = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
VGG_PATH = "imagenet-vgg-verydeep-19.mat"
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 300

CONTENT_LAYERS = [('conv4_2', 0.5), ('conv5_2', 0.5)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

NOISE_RATIO = 0.5
MEAN_PIXELS = [128, 128, 128]

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 100

EPOCHES = 500