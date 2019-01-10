import sys
import os
import math
import six
import tensorflow as tf

sys.path.append(os.getcwd() + '/tf_models/research')
sys.path.append(os.getcwd() + '/tf_models/research/slim')
try:
    from deeplab import common
    from deeplab import model
    from deeplab_overrides.datasets import segmentation_dataset
    from deeplab.utils import input_generator
except:
    print('Can\'t import deeplab libraries!')
    raise

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', './logs/eval',
                    'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', './logs',
                    'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [225, 225],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 2,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', True,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'coco',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', './data/records',
                    'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        tf.local_variables_initializer()
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return tf.reduce_mean(tf.stack(prec), axis=0)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get dataset-dependent information.
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

    tf.gfile.MakeDirs(FLAGS.eval_logdir)
    tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

    with tf.Graph().as_default():
        samples = input_generator.get(
            dataset,
            FLAGS.eval_crop_size,
            FLAGS.eval_batch_size,
            min_resize_value=224,
            max_resize_value=224,
            resize_factor=FLAGS.resize_factor,
            dataset_split=FLAGS.eval_split,
            is_training=False,
            model_variant=FLAGS.model_variant)

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
            crop_size=FLAGS.eval_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')
            logits, predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                               image_pyramid=FLAGS.image_pyramid)
        else:
            tf.logging.info('Performing multi-scale test.')
            predictions = model.predict_labels_multi_scale(
                samples[common.IMAGE],
                model_options=model_options,
                eval_scales=FLAGS.eval_scales,
                add_flipped_images=FLAGS.add_flipped_images)
        predictions = predictions[common.OUTPUT_TYPE]
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.
        labels = tf.where(
            tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = 'miou'
        for eval_scale in FLAGS.eval_scales:
            predictions_tag += '_' + str(eval_scale)
        if FLAGS.add_flipped_images:
            predictions_tag += '_flipped'

        # logits = tf.reshape(logits, shape=[50625, 3])
        # thresholds = tf.concat(
        #     [tf.constant(0.9, dtype=tf.float32, shape=[225 * 225, 1]), logits], axis=1)
        # logits = tf.arg_max(thresholds, dimension=1)

        # # Define the evaluation metric.
        # metric_map = {}
        # metric_map[predictions_tag+'.9'] = tf.metrics.mean_iou(
        #     logits, labels, 4, weights=weights)

        logits = tf.reshape(tf.reduce_max(logits, axis=[-1]), shape=[-1])

        # Define the evaluation metric.
        metric_map = {}
        metric_map[predictions_tag+'.5'] = tf.metrics.mean_iou(
            tf.to_int32(logits > 0.5), tf.to_int32(labels > 0), 2, weights=weights)
        metric_map[predictions_tag+'.9'] = tf.metrics.mean_iou(
            tf.to_int32(logits > 0.9), tf.to_int32(labels > 0), 2, weights=weights)
        metric_map[predictions_tag+'.95'] = tf.metrics.mean_iou(
            tf.to_int32(logits > 0.95), tf.to_int32(labels > 0), 2, weights=weights)
        metric_map[predictions_tag+'.98'] = tf.metrics.mean_iou(
            tf.to_int32(logits > 0.98), tf.to_int32(labels > 0), 2, weights=weights)
        metric_map[predictions_tag+'.99'] = tf.metrics.mean_iou(
            tf.to_int32(logits > 0.99), tf.to_int32(labels > 0), 2, weights=weights)

        # def mean_iou(y_true, y_pred, weights, t):
        #     y_pred_ = tf.cast(y_pred > t, tf.int32)
        #     score, up_opt = tf.metrics.mean_iou(
        #         y_true, y_pred_, 2, weights=weights)
        #     return score, up_opt

        # flatten_logits = tf.reshape(logits, shape=[-1, 4])
        # one_hot_labels = slim.one_hot_encoding(
        #     labels, 4, on_value=1.0, off_value=0.0)
        # metric_map['m0.05'] = mean_iou(
        #     one_hot_labels, flatten_logits, weights, 0.05)
        # metric_map['m0.5'] = mean_iou(
        #     one_hot_labels, flatten_logits, weights, 0.5)
        # metric_map['m1.0'] = mean_iou(
        #     one_hot_labels, flatten_logits, weights, 1.0)

        metrics_to_values, metrics_to_updates = (
            tf.contrib.metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in six.iteritems(metrics_to_values):
            slim.summaries.add_scalar_summary(
                metric_value, metric_name, print_summary=True)

        num_batches = int(
            math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

        tf.logging.info('Eval num images %d', dataset.num_samples)
        tf.logging.info('Eval batch size %d and num batch %d',
                        FLAGS.eval_batch_size, num_batches)

        num_eval_iters = None
        if FLAGS.max_number_of_evaluations > 0:
            num_eval_iters = FLAGS.max_number_of_evaluations
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logdir=FLAGS.eval_logdir,
            num_evals=num_batches,
            eval_op=list(metrics_to_updates.values()),
            max_number_of_evaluations=num_eval_iters,
            eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('eval_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
