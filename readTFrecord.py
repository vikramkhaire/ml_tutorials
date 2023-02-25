import tensorflow as tf
import numpy as np

# for testing
filename = "/home/vikram/ml_tutorials/tfrecord/train-00002-of-00008"
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)


def load_tfrecord_variable(serialized_example):
    context_features = {
        'kepid': tf.io.FixedLenFeature([], dtype=tf.int64),
        'av_pred_class': tf.io.FixedLenFeature([], dtype=tf.string)
    }

    sequence_features = {
        "local_view": tf.io.VarLenFeature(tf.int64)
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    kpeid = context_parsed['kpeid']
    av_pred_class = context_parsed['av_pred_class']

    local_view = sequence_parsed['local_view'].values

    return tf.tuple([kpeid, av_pred_class, local_view])

x = load_tfrecord_variable(raw_dataset)
print(x)

"""
def decode_fn(record_bytes):
    return tf.io.parse_single_example(
    record_bytes,
    # Schema
    {"kepid": tf.io.FixedLenFeature([], dtype=tf.int64),
     "local_view": tf.io.FixedLenFeature([[array.shape[0]], dtype=tf.float64)}
    )

for batch in tf.data.TFRecordDataset([filename]).map(decode_fn):
    #print("kepid = {kepid:.4f},  tce_depth = {tce_depth:.4f}".format(**batch))
    print(kepid, local_view)
"""



"""
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


result = {}
# example.features.feature is the dictionary
for key, feature in example.features.feature.items():
  # The values are the Feature objects which contain a `kind` which contains:
  # one of three fields: bytes_list, float_list, int64_list

  kind = feature.WhichOneof('kind')
  result[key] = np.array(getattr(feature, kind).value)

print(result)

for key in list(result.keys()):
    print(key, "------------------------")
    print(result[key])


"""
