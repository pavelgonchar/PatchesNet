from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count, splits=None):
    def get_slice(data, share, total, start):
        shape = tf.shape(data)
        print(share, total, start, shape)

        _size = tf.concat([ share * shape[:1] // total, shape[1:] ],axis=0)
        #stride = tf.concat([ share * shape[:1] // total, shape[1:]*0 ],axis=0)
        _start = tf.concat([ start * shape[:1] // total, shape[1:]*0 ],axis=0)
        print(_size, _start)
        return tf.slice(data, _start, _size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    start = 0

    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'share':splits[i],'total' : sum(splits), 'start':start})(x)
                    start += splits[i]
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        ##return Model(input=model.inputs, output=merged)
	new_model = Model(input=model.inputs, output=merged)
	funcType = type(model.save)

	# monkeypatch the save to save just the underlying model
	def new_save(self_,filepath, overwrite=True):
    		model.save(filepath, overwrite)
	new_model.save=funcType(new_save, new_model)
	return new_model
