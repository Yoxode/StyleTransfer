import config
import vgg
import tensorflow as tf
import numpy as np
import utils
import scipy.misc


def gram_matrix(f, n, m):
    """
    get gram matrix
    :param f: input tensor
    :param n: size of image (width * height)
    :param m: number of channels
    :return: gram matrix of tensor f
    """
    f = tf.reshape(f, (m, n))
    return tf.matmul(tf.transpose(f), f)


def total_loss(sess, model):
    content_layers = config.CONTENT_LAYERS
    sess.run(tf.assign(model.model['input'], model.content))
    content_loss = 0.0
    for layer_name, weight in content_layers:
        p = sess.run(model.model[layer_name])
        x = model.model[layer_name]
        M = p.shape[1] * p.shape[2]
        N = p.shape[3]
        content_loss += (1.0 / (2 * M * N)) * tf.reduce_sum(tf.pow(p - x, 2)) * weight
    content_loss /= len(content_layers)
    style_layers = config.STYLE_LAYERS
    sess.run(tf.assign(model.model['input'], model.style))
    style_loss = 0.0
    for layer_name, weight in style_layers:
        a = sess.run(model.model[layer_name])
        x = model.model[layer_name]
        M = a.shape[1] * a.shape[2]
        N = a.shape[3]
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        style_loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * weight
    style_loss /= len(style_layers)
    loss = config.CONTENT_WEIGHT * content_loss + config.STYLE_WEIGHT * style_loss

    return loss


def train():
    model = vgg.VGG(config.CONTENT_PATH, config.STYLE_PATH)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cost = total_loss(sess, model)
        optimizer = tf.train.AdamOptimizer(1.0).minimize(cost)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.model['input'], model.noise_image))
        for step in range(config.EPOCHES):
            sess.run(optimizer)
            if step % 50 == 0:
                print('step {} is down.'.format(step))
                img = sess.run(model.model['input'])
                img += config.MEAN_PIXELS
                img = img[0]
                img = np.clip(img, 0, 255).astype(np.uint8)
                scipy.misc.imsave('{}output-{}.jpg'.format(config.OUTPUT_PATH, step), img)
        img = sess.run(model.model['input'])
        img += config.MEAN_PIXELS
        img = img[0]
        img = np.clip(img, 0, 255).astype(np.uint8)
        scipy.misc.imsave('{}.jpg'.format(config.OUTPUT_PATH), img)


def main():
    utils.download_vgg(config.VGG_URL, config.VGG_PATH)
    utils.make_dir(config.OUTPUT_PATH)
    train()


if __name__ == "__main__":
    main()