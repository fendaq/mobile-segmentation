import tensorflow as tf

def dice_coefficient(logits, target, smooth=1.0):
    logits = tf.reshape(logits, shape=[-1])
    target = tf.reshape(target, shape=[-1])
    intersection = tf.reduce_sum(logits * target)
    return (intersection + smooth) / (tf.reduce_sum(logits) + tf.reduce_sum(target) - intersection + smooth)
