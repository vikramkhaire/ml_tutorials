import tensorflow as tf
import numpy as np
import astropy.table as tab
import pandas_tfrecords as pdtfr

def get_example(raw_dataset, take_val = 1):
    for raw_record in raw_dataset.take(take_val):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        #print(example)

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

"""
# first get an example
result = get_example(raw_dataset=raw_dataset)
key_list = sorted(list(result.keys()))
"""

key_list = ['av_pred_class','av_training_set', 'global_view', 'kepid', 'local_view', 'spline_bkspace', 'tce_depth',
            'tce_duration','tce_impact','tce_max_mult_ev', 'tce_model_snr', 'tce_period', 'tce_plnt_num', 'tce_prad', 'tce_time0bk']

av_pred_class = []
av_training_set = []
global_view =[]
kepid =[]
local_view =[]
spline_bkspace = []
tce_depth = []
tce_duration =[]
tce_impact =[]
tce_max_mult_ev =[]
tce_model_snr =[]
tce_period =[]
tce_plnt_num =[]
tce_prad =[]
tce_time0bk =[]

# hack to find length of this file
pdrecord = pdtfr.tfrecords_to_pandas('/home/vikram/ml_tutorials/tfrecord/train-00002-of-00008', schema ={'kepid':int, 'tce_period': float})

for i in range(5):#len(pdrecord)):
    result = get_example(raw_dataset=raw_dataset, take_val= i+1)
    print(i)
    av_pred_class.append(str(result['av_pred_class'][0], 'utf-8'))
    av_training_set.append(result['av_training_set'][0])
    global_view.append(result['global_view'])
    kepid.append(result['kepid'][0])
    local_view.append(result['local_view'])
    spline_bkspace.append(result['spline_bkspace'][0])
    tce_depth.append(result['tce_depth'][0])
    tce_duration.append(result['tce_duration'][0])
    tce_impact.append(result['tce_impact'][0])
    tce_max_mult_ev.append(result['tce_max_mult_ev'][0])
    tce_model_snr.append(result['tce_model_snr'][0])
    tce_period.append(result['tce_period'][0])
    tce_plnt_num.append(result['tce_plnt_num'][0])
    tce_prad.append(result['tce_prad'][0])
    tce_time0bk.append(result['tce_time0bk'][0])



