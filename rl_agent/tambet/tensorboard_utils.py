import tensorflow as tf
import numpy as np

def create_summary_writer(logdir):
    return tf.summary.FileWriter(logdir)


def create_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

# https://stackoverflow.com/questions/42012906/create-a-custom-tensorflow-histogram-summary
def create_histogram(tag, values, bins=100):
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Return Summary
    return tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])

def add_summary(writer, tag, value, step):
    writer.add_summary(create_summary(tag, value), global_step=step)


def add_histogram(writer, tag, values, step, bins=100):
    writer.add_summary(create_histogram(tag, values, bins), global_step=step)


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        s = create_summary(prefix + t, v)
        writer.add_summary(s, global_step=step)

