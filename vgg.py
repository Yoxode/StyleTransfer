import config
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc


class VGG(object):

    def __init__(self, content_path=config.CONTENT_PATH, style_path=config.STYLE_PATH):

        self.content = self.load(content_path)

        self.style = self.load(style_path)

        self.noise_image = self.gen_noise()

        self.model = self.vgg_model()

    def vgg_model(self):

        vgg = scipy.io.loadmat(config.VGG_PATH)

        vgg_layers = vgg['layers'][0]

        model = dict()

        model['input'] = tf.Variable(np.zeros([1, config.IMAGE_HEIGHT, config.IMAGE_WIDTHM, 3]), dtype=tf.float32)

        model['conv1_1'] = self.forward(model['input'], self.get_wb(vgg_layers, 0))
        model['conv1_2'] = self.forward(model['conv1_1'], self.get_wb(vgg_layers, 2))
        model['pool1'] = self.max_pool(model['conv1_2'])
        model['conv2_1'] = self.forward(model['pool1'], self.get_wb(vgg_layers, 5))
        model['conv2_2'] = self.forward(model['conv2_1'], self.get_wb(vgg_layers, 7))
        model['pool2'] = self.max_pool(model['conv2_2'])
        model['conv3_1'] = self.forward(model['pool2'], self.get_wb(vgg_layers, 10))
        model['conv3_2'] = self.forward(model['conv3_1'], self.get_wb(vgg_layers, 12))
        model['conv3_3'] = self.forward(model['conv3_2'], self.get_wb(vgg_layers, 14))
        model['conv3_4'] = self.forward(model['conv3_3'], self.get_wb(vgg_layers, 16))
        model['pool3'] = self.max_pool(model['conv3_4'])
        model['conv4_1'] = self.forward(model['pool3'], self.get_wb(vgg_layers, 19))
        model['conv4_2'] = self.forward(model['conv4_1'], self.get_wb(vgg_layers, 21))
        model['conv4_3'] = self.forward(model['conv4_2'], self.get_wb(vgg_layers, 23))
        model['conv4_4'] = self.forward(model['conv4_3'], self.get_wb(vgg_layers, 25))
        model['pool4'] = self.max_pool(model['conv4_4'])
        model['conv5_1'] = self.forward(model['pool4'], self.get_wb(vgg_layers, 28))
        model['conv5_2'] = self.forward(model['conv5_1'], self.get_wb(vgg_layers, 30))
        model['conv5_3'] = self.forward(model['conv5_2'], self.get_wb(vgg_layers, 32))
        model['conv5_4'] = self.forward(model['conv5_3'], self.get_wb(vgg_layers, 34))
        model['pool5'] = self.max_pool(model['conv5_4'])

        return model

    def forward(self, inputs, wb):
        conv_res = tf.nn.conv2d(inputs, wb[0], [1, 1, 1, 1], padding='SAME')
        relu_res = tf.nn.relu(conv_res + wb[1])
        return relu_res

    def max_pool(self, inputs):

        return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    def get_wb(self, layers, num_layer):

        w = layers[num_layer][0][0][2][0][0]
        b = layers[num_layer][0][0][2][0][1]
        return w, b.reshape(b.size)

    def gen_noise_image(self):
        noise = np.random.uniform(-20, 20, [1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])
        noise_image = noise * config.NOISE_RATIO + self.content * (1 - config.NOISE_RATIO)
        return noise_image

    def load_image(self, path):

        image = scipy.misc.imread(path)

        image = scipy.misc.imresize(image, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])

        image = np.reshape(image, (1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))

        image = image - config.MEAN_PIXELS

        return image


if __name__ == '__main__':
    VGG()
