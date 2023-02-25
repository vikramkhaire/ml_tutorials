import tensorflow as tf
import numpy as np
import astropy.table as tab


def get_example(raw_dataset, take_val = 1):
    for raw_record in raw_dataset.take(take_val):
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

    return result


# for testing
filename = "/home/vikram/ml_tutorials/tfrecord/train-00002-of-00008"
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)


# first get an example
result = get_example(raw_dataset=raw_dataset)



for key in list(result.keys()):
    print(key, "------------------------")
    print(result[key])


data = tab.Table()



