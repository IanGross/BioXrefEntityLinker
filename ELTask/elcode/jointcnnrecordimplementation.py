from random import shuffle

import numpy as np
import tensorflow as tf

import data_constants
import pickle

EMBEDFILE_NAME = "../dumps/word_embeddings.pkl"
EMBEDDING_SIZE = 250

# embedding_fname is numpy matrix of embeddings (V x d) where V is vocab size and d is embedding dim
def _load_embeddings(embedding_fname):
    with open(embedding_fname, 'rb') as pe_f:
        pretrained_embed = pickle.load(pe_f)
    return pretrained_embed.shape, tf.constant_initializer(pretrained_embed)

def _parse_el_example(array_feats, array_feat_types, quant_feats):
    """
        Parse a single serialized example from a tfrecord file.
    """
    out_example = []
    d_keys = sorted(array_feats.keys())
    for k in d_keys:
        n_feat = quant_feats[k]
        point_feat = tf.decode_raw(array_feats[k], array_feat_types[k])
        point_feat = tf.reshape(point_feat, [quant_feats[k]])
        out_example.append(point_feat)
    return tuple(out_example)


def get_data_itr(**kwargs):
    """
        Return a data iterator for the tfrecord files.
    """
    batch_size = kwargs.pop('batch_size', 4)
    num_threads = kwargs.pop('num_threads', 8)
    fnames = kwargs.pop('fnames', [])
    q_capacity = kwargs.pop('q_capacity', 256)
    shuff = kwargs.pop('shuffle_input', True)
    serialized_keys = kwargs.pop('serialized_keys', data_constants.READER_FEAT_KEY_TYPE_MAP)
    out_type_keys = kwargs.pop('out_type_keys', data_constants.TF_FEAT_KEY_TYPE_MAP)
    feature_shapes = kwargs.pop('feature_shapes', data_constants.DATA_SHAPES)

    quantity_keys = {key: quant_type for key, quant_type in serialized_keys.items() if key[:2] == 'n_'}
    arr_key_type = {key: quant_type for key, quant_type in out_type_keys.items() if key[:2] != 'n_'}

    with tf.name_scope('input'):
        with tf.device('/cpu:0'):
            features_dict = {k: tf.FixedLenFeature((), t) for k, t in serialized_keys.items()}

            def _parser(serialized_example):
                features = tf.parse_single_example(serialized_example,
                                                   features=features_dict)
                quant_features = {}
                for k, t in quantity_keys.items():
                    quant_features[k[2:]] = tf.to_int32(features[k])

                arr_features = {}
                for k, t in arr_key_type.items():
                    arr_features[k] = features[k]

                exam = _parse_el_example(arr_features,
                                         arr_key_type,
                                         quant_features)
                return exam

            dataset = tf.data.TFRecordDataset(fnames)

            if num_threads > 1:
                dataset = dataset.map(_parser, num_threads=num_threads)
            else:
                dataset = dataset.map(_parser)

            if shuff:
                dataset = dataset.shuffle(buffer_size=q_capacity)

            ordered_arr_key_type = sorted(arr_key_type.keys())
            pad_shape = tuple([feature_shapes[arr_k] for arr_k in ordered_arr_key_type])

            dataset = dataset.padded_batch(batch_size=batch_size,
                                           padded_shapes=pad_shape)
            itr = dataset.make_initializable_iterator()
            return itr, itr.get_next()


if __name__ == '__main__':
    def F(a, b):
        return a, b

    #Defining helper functions for forward prop
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    #Defining a template function to reuse for 2x2 pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    '''
    CNN with multiple filter sizes and convolves over same input
    '''
    def mul_filtercnn(filter_sizes, input_data):
        #Need the maximum sequence length amongst the batch
        #sequence_length = 0
        #sequence_length = input_data.get_shape().as_list()
        #print("sequence_length ", input_data.shape)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters = 2
        print("Shape of input data ", input_data.shape)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(\
                  inputs=input_data,\
                  filters=num_filters,\
                  kernel_size=[filter_size,EMBEDDING_SIZE],\
                  padding="VALID",\
                  activation=tf.nn.sigmoid)
                '''
                Applying 6 filters, so these should give two filter values for each filter size?
                Batch, sen_len, dim, 2
                Pooling needs to choose the maximal value from each of these two filters
                '''
                print(" Shape of conv ",conv.shape)
                # Apply nonlinearity
                # Maxpooling over the outputs
                #pooled = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
                # Max-pooling over the outputs
                # pooled = tf.nn.max_pool(
                #     conv,
                #     ksize=[1, 14 - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID')
                pooled = tf.reduce_max(conv, axis=[1])
                print("Pooled shape ",pooled.shape)
                pooled_outputs.append(pooled)
        print(" Pooled ",pooled_outputs)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs,axis=2)
        h_pool_flat = tf.expand_dims(tf.reshape(h_pool, [-1, num_filters_total]),-1)
        print("Shape of h_pool final ",h_pool_flat)
        return h_pool_flat

    '''
    For sentence network - 2 Conv[12 (3 x 3), 12 (3 x 3)], 1 Pool , Fully dense 5112
    For ont network - 3 Conv [50 (7x7), 50 (5x5), 75 (3 x 3)], 2 Pool, Fully dense 5112
    Both will share logit
    loss = max(cos(fc1,fc2))
    Generate the output value given the weight of the last ouput layer
    and data(either training or testing)
    The softmax method will be applied on this matrix

    '''
    def model(sent_data,ont_data):
        sent_data = tf.expand_dims(sent_data, -1)
        ont_data = tf.expand_dims(ont_data, -1)
        """
        Assembles the output function
        """

        # # Convolutional Layer #1
        # conv1 = tf.layers.conv2d(
        #       inputs=input_layer,
        #       filters=12,
        #       kernel_size=[3, 3],
        #       padding="same",
        #       activation=tf.nn.relu)

        # # Pooling Layer #1
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        #Forward prop phase for sent network
        # c1_sent = tf.identity(tf.layers.conv2d(\
        #       inputs=sent_data,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c1_sent")
        # p1_sent = tf.layers.max_pooling2d(inputs=c1_sent, pool_size=[2, 2], strides=2)
        # c2_sent = tf.identity(tf.layers.conv2d(\
        #       inputs=p1_sent,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c2_sent")
        # c3_sent = tf.identity(tf.layers.conv2d(\
        #       inputs=c2_sent,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c3_sent")
        #c4_sent = tf.identity(tf.nn.relu(conv2d(c2_sent, w_sentc4)),name="c4_sent")
        filter_sizes = [2,3, 5]
        filter_bitsent = mul_filtercnn(filter_sizes, sent_data)
        #filter_sizesp2 = [3,3,3]
        #pool2 = mul_filtercnn(filter_sizes, pool1)
        #pool3 = mul_filtercnn(filter_sizes, pool2)
        #flat_sent = tf.identity(tf.layers.Flatten()(pool1),name="flat_sent")
        #flat_sentex = tf.expand_dims(flat_sent,-1)
        fc_sent = tf.identity(tf.layers.conv1d(\
              inputs=filter_bitsent,\
              filters=1,\
              kernel_size=1,\
              padding="same",\
              activation=tf.nn.sigmoid),name="fc_sent")
        #Add logits and softmax here

        #Forward prop phase for ont network,change this to accept just the ont alone?
        # c1_ont = tf.identity(tf.layers.conv2d(\
        #       inputs=ont_data,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c1_ont")
        # p1_ont = tf.layers.max_pooling2d(inputs=c1_ont, pool_size=[2, 2], strides=2)
        # c2_ont = tf.identity(tf.layers.conv2d(\
        #       inputs=p1_ont,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c2_ont")
        # p2_ont = tf.layers.max_pooling2d(inputs=c2_ont, pool_size=[2, 2], strides=2)
        # c3_ont = tf.identity(tf.layers.conv2d(\
        #       inputs=p2_ont,\
        #       filters=1,\
        #       kernel_size=[3, 250],\
        #       padding="same",\
        #       activation=tf.nn.relu),name="c3_ont")
        filter_sizesont = [3, 5, 7]
        filter_bitont = mul_filtercnn(filter_sizesont, ont_data)
        #pool2ont = mul_filtercnn(filter_sizes, pool1ont)
        #pool3ont = mul_filtercnn(filter_sizes, pool2ont)
        #flat_ont = tf.identity(tf.layers.Flatten()(pool1ont),name="flat_ont")
        #flat_ontex = tf.expand_dims(flat_ont,-1)
        fc_ont = tf.identity(tf.layers.conv1d(\
              inputs=filter_bitont,\
              filters=1,\
              kernel_size=1,\
              padding="same",\
              activation=tf.nn.sigmoid),name="fc_ont")
        return (fc_sent, fc_ont)

    '''
    Calculate the cosine similarity between ont rep and sent rep
    Calc loss for every data point, and take the mean
    '''
    def calc_loss(sent_rep,ont_rep):
        normalised_sent = tf.nn.l2_normalize(sent_rep,dim=2)
        normalised_ont = tf.nn.l2_normalize(ont_rep,dim=2)
        #Averaging out the loss
        #return tf.losses.cosine_distance(normalised_sent,normalised_ont,dim=None,weights=1.0,scope=None,axis=1,loss_collection=tf.GraphKeys.LOSSES,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        #return tf.losses.cosine_distance(normalised_sent,normalised_ont,dim=None,scope=None,axis=1)
        #This is going completely wrong
        return 1 - tf.reduce_mean(tf.matmul(normalised_sent,normalised_ont,transpose_b=True))

    '''
    Accuracy is computed by comparing the predicted and actual labels
    argmax is used on the 1-K encoded arrays for the same
    Then the mean value of the number of matches is returned
    Since the softmax is not applied during the training phase a softmax needs to be applied before checking for accuracy
    '''
    def accuracy(preds, expected):       
        correct_prediction = tf.equal(tf.argmax(preds, 1),tf.argmax(expected, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        return accuracy*100


    '''
    Find per class Classification Error
    Find the argmax and compare the indices of the argmax at a per class level
    Returns a dictionary of the losses against digits
    This will later be leveraged to find Classification errors for each class
    '''
    def calcerror_perclass(predictions,labels):
        digits = {i:{"val":0,"size":0} for i in range(0,1827)}
        #Need to apply a softmax during the testing step
        preds_arg = predictions
        labels_np = labels
        #Converting both of them to np.array, so that it is easy loop and store values
        preds_labels = np.argmax(preds_arg,1)
        labels_indices = np.argmax(labels_np,1)
        preds_size = len(preds_labels)
        print(" Pred size ", preds_size)

        print("Shape of preds labels",preds_labels.shape,"Shape of labels",labels_indices.shape,"Size of actual array",preds_size)
        for pred_index in range(0,preds_size):
            #If the actual label and the predicted label don't match, then add one against the actual label
            if labels_indices[pred_index] != preds_labels[pred_index]:
                #print("Val is equal",pred_index)
                digits[labels_indices[pred_index]]["val"] += 1
            #Divide by all mentions
            digits[labels_indices[pred_index]]["size"] += 1
        for key in digits:
            #No data for a particular class?
            if digits[key]["size"] != 0:
                digits[key]["val"] = (digits[key]["val"]/digits[key]["size"])*100
            else:
                digits[key]["val"] = 150
        return digits


    batch_n = 40
    n_threads = 1
    fname_list = ['../dumps/train.data0.tfrecord', '../dumps/train.data1.tfrecord']
    fname_testlist = ['../dumps/test.data0.tfrecord', '../dumps/test.data1.tfrecord']
    fname_holder = tf.placeholder(tf.string, shape=[None])
    #Google what this means?
    buff_size = 2
    shuff_data = True
    serial_keys = data_constants.READER_FEAT_KEY_TYPE_MAP
    output_type_keys = data_constants.TF_FEAT_KEY_TYPE_MAP
    feat_dims = data_constants.DATA_SHAPES

    n_epochs = 4


    itr, next_elem = get_data_itr(batch_size=batch_n,
                                  num_threads=n_threads,
                                  fnames=fname_holder,
                                  q_capacity=buff_size,
                                  shuffle_input=shuff_data,
                                  serialized_keys=serial_keys,
                                  out_type_keys=output_type_keys,
                                  feature_shapes=feat_dims
                                )

    embed_shape, embed_init = _load_embeddings(EMBEDFILE_NAME)
    E = tf.get_variable('embedding_layer', shape=embed_shape, initializer=embed_init)

    embed_tf_worddataset = tf.nn.embedding_lookup(E, next_elem[0])
    #1 - GO:0000 , GO:000 - [12,24]
    embed_tf_ontdataset = tf.nn.embedding_lookup(E, next_elem[1])

    #Need to add code for tf.reshape here
    O = model(embed_tf_worddataset, embed_tf_ontdataset)
    #Loss per batch is calculated as the cosine distance between the sentence and ontology representation
    loss=calc_loss(O[0],O[1])

    #Run the Adam Optimiser(AdaGrad + Momentum) with an initial eta of 0.0001
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        seen_feats = []
        count = 0

        for _ in range(n_epochs):
            print(" Starting an epoch ")
            shuffle(fname_list)
            sess.run(itr.initializer, feed_dict={fname_holder: fname_list})
            while True:
                try:
                    #Initialise the embeddings
                    sess.run(E)
                    (x, y), cos_dist, _, z = sess.run([O, loss,train_step, next_elem[2]])
                    #print(" Sent embedding ", s)
                    print(" Description ",x.shape)
                    print("Sent ", y.shape)
                    print(" Cosine distance is ", cos_dist)
                    print("Truth ", z.shape)
                except tf.errors.OutOfRangeError:
                    print('Completed epoch')
                    break
        sess.close()
