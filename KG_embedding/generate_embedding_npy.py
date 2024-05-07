import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
from tensorflow.python import pywrap_tensorflow
import numpy as np
# with tf.Session() as sess:
#   new_saver = tf.train.import_meta_graph('/data/c_x/joie-kdd19/model/models/transe_CMP-linear_dim1_300_dim2_100_a1_2.5_a2_1.0_m1_0.5_fold_3/transe-model-m2.ckpt.meta')
#   new_saver.restore(sess, tf.train.latest_checkpoint('/data/c_x/joie-kdd19/model/models/transe_CMP-linear_dim1_300_dim2_100_a1_2.5_a2_1.0_m1_0.5_fold_3'))
#   print([tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
# )

# model_exp = "/data/c_x/joie-kdd19/model/models"
model_exp = "/data/c_x/joie-kdd19/model/bioschs_202308152116/hole_CMP-linear_dim1_300_dim2_100_a1_2.5_a2_1.0_m1_0.5_fold_3"
#model_exp = "model-lenet"

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        #raise load_modelValueError('No meta file found in the model directory (%s)' % model_dir)
        print("No meta file found in the model directory ")
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


meta_file, ckpt_file = get_model_filenames(model_exp)
data=open("data.txt",'w+')
print('Metagraph file: %s' % meta_file, file=data) 
print('Checkpoint file: %s' % ckpt_file, file=data)
print('Metagraph file: %s' % meta_file)
print('Checkpoint file: %s' % ckpt_file)

reader = tf.train.NewCheckpointReader(os.path.join(model_exp, ckpt_file))
var_to_shape_map = reader.get_variable_to_shape_map()
name = ['graph/ht2', 'graph/ht1', 'graph/r2', 'graph/r1']
for key in var_to_shape_map:
    if key in name:
        print(type(reader.get_tensor(key)))
        np.save('/data/c_x/joie-kdd19/embeddings/bios_chs/{}.npy'.format(key.replace('/','-')), reader.get_tensor(key))

data.close()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
    saver.restore(tf.get_default_session(),
                  os.path.join(model_exp, ckpt_file))
    #print(tf.get_default_graph().get_tensor_by_name("Logits/weights:0"))

