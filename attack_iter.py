from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy
from scipy import misc
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from nets import inception_v1, vgg, resnet_v1
import cv2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v1', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 66, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):           #modified   origin png
    image = cv2.imread(filepath, 1).astype(np.float) / 255.0
    image = cv2.resize(image, (224,224))
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    scipy.misc.imsave(os.path.join(output_dir, filename), (images[i, :, :, :] + 1.0) * 0.5)

def graph(x, y, i, x_max, x_min, grad):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  alpha = eps / num_iter
  momentum = FLAGS.momentum
  num_classes = 110             #modified    origin 1001

  with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    logits_v1, end_points_v1 = inception_v1.inception_v1(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_resnet, end_points_resnet = resnet_v1.resnet_v1_50(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg, end_points_vgg = vgg.vgg_16(
        x, num_classes=num_classes, is_training=False)           #modified


  pred = tf.argmax(end_points_v1['Predictions'] + end_points_resnet['Predictions'] + end_points_vgg['Predictions'], 1)  #modified


  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)


  logits = (logits_v1 + logits_vgg + logits_resnet) / 3

  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)

  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


def main(_):
  eps = 2.0 * FLAGS.max_epsilon / 255.0

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))


    with tf.Session() as sess:
      s1.restore(sess, FLAGS.checkpoint_path_inception_v1)
      s2.restore(sess, FLAGS.checkpoint_path_resnet)
      s3.restore(sess, FLAGS.checkpoint_path_vgg)


      for filenames, images in load_images(FLAGS.input_dir, (1, 224, 224, 3)):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()
