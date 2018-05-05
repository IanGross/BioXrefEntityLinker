import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import json

'''
Define the graph which specifies placeholders, ops and trainable variables
'''
graph = tf.Graph()
with graph.as_default():

    initializer = tf.contrib.layers.xavier_initializer()

    # '''
    # Initialise weights based on Xavier initialisation
    # Dimensions of weight go by prev layer feeding into current layer 
    # For convolutional layer - [patch,patch,no_of_channels,no_of_filters]
    # For dense layer - [prev_layer, next_layer]
    # '''
    # initializer = tf.contrib.layers.xavier_initializer()
    # w_sentc1 = tf.Variable(initializer([3,3,1,12]),name="w_sentc1")  
    # w_sentc2 = tf.Variable(initializer([3,3,12,12]),name="w_sentc2")
    # w_sentc3 = tf.Variable(initializer([3,3,12,12]),name="w_sentc3")
    # #w_sentc4 = tf.Variable(initializer([3,3,12,12]),name="w_sentc4")

    # w_ontc1 = tf.Variable(initializer([7,7,1,50]),name="w_ontc1")  
    # w_ontc2 = tf.Variable(initializer([5,5,50,50]),name="w_ontc2")
    # w_ontc3 = tf.Variable(initializer([3,3,50,75]),name="w_ontc3")


    '''Feed in batches of input data, the None parameter allows us to have variable number of data points fed into the batch
    The input is a video this time round, so the dimensions are (batch_size, no_frames, img_width, img_height, channels)
    '''
    tf_worddataset = tf.placeholder(tf.float32, shape=(None,12,250,1),name="tf_dataset")
    tf_ontdataset = tf.placeholder(tf.float32, shape=(None,50,250,1),name="tf_ontdataset")
    tf_labels = tf.placeholder(tf.float32, shape=(None, 1827),name="tf_labels")
    batch_size = 100

    #Defining helper functions for forward prop
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    #Defining a template function to reuse for 2x2 pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

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
        c1_sent = tf.identity(tf.layers.conv2d(\
              inputs=sent_data,\
              filters=12,\
              kernel_size=[3, 3],\
              padding="same",\
              activation=tf.nn.relu),name="c1_sent")
        p1_sent = tf.layers.max_pooling2d(inputs=c1_sent, pool_size=[2, 2], strides=2)
        c2_sent = tf.identity(tf.layers.conv2d(\
              inputs=p1_sent,\
              filters=12,\
              kernel_size=[3, 3],\
              padding="same",\
              activation=tf.nn.relu),name="c2_sent")
        c3_sent = tf.identity(tf.layers.conv2d(\
              inputs=c2_sent,\
              filters=12,\
              kernel_size=[3, 3],\
              padding="same",\
              activation=tf.nn.relu),name="c3_sent")
        #c4_sent = tf.identity(tf.nn.relu(conv2d(c2_sent, w_sentc4)),name="c4_sent")
        flat_sent = tf.identity(tf.layers.Flatten()(c3_sent),name="flat_sent")
        fc_sent = tf.identity(tf.layers.dense(inputs=flat_sent, units=5112, activation=tf.nn.relu),name="dense_sent")
        #Add logits and softmax here

        #Forward prop phase for ont network
        c1_ont = tf.identity(tf.layers.conv2d(\
              inputs=ont_data,\
              filters=50,\
              kernel_size=[7, 7],\
              padding="same",\
              activation=tf.nn.relu),name="c1_ont")
        p1_ont = tf.layers.max_pooling2d(inputs=c1_ont, pool_size=[2, 2], strides=2)
        c2_ont = tf.identity(tf.layers.conv2d(\
              inputs=p1_ont,\
              filters=50,\
              kernel_size=[5, 5],\
              padding="same",\
              activation=tf.nn.relu),name="c2_ont")
        p2_ont = tf.layers.max_pooling2d(inputs=c2_ont, pool_size=[2, 2], strides=2)
        c3_ont = tf.identity(tf.layers.conv2d(\
              inputs=p2_ont,\
              filters=75,\
              kernel_size=[3, 3],\
              padding="same",\
              activation=tf.nn.relu),name="c3_ont")
        flat_ont = tf.identity(tf.layers.Flatten()(c3_ont),name="flat_ont")
        fc_ont = tf.identity(tf.layers.dense(inputs=flat_ont, units=5112, activation=tf.nn.relu),name="dense_ont")   
        return (fc_sent, fc_ont)

    '''
    Calculate the cosine similarity between ont rep and sent rep
    Calc loss for every data point, and take the mean
    '''
    def calc_loss(sent_rep,ont_rep):
        normalised_sent = tf.nn.l2_normalize(sent_rep,dim=1)
        normalised_ont = tf.nn.l2_normalize(ont_rep,dim=1)
        #Averaging out the loss
        return tf.losses.cosine_distance(normalised_sent,normalised_ont,dim=None,weights=1.0,scope=None,axis=0,loss_collection=tf.GraphKeys.LOSSES,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        #return 1- tf.reduce_mean(tf.matmul(normalised_sent,normalised_ont,transpose_b=True))
        #return tf.reduce_mean(tf.multiply(tf.nn.l2_normalize(ont_rep,0), tf.nn.l2_normalize(sent_rep,0)))

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

    #Get prediction on training batch, this is the non-argmaxed and non-softmaxed one
    (fc_sent, fc_ont) = model(tf_worddataset,tf_ontdataset)

    prediction = tf.identity(tf.nn.softmax(tf.layers.dense(inputs=fc_sent, units=1827)),name="pred_op")
    accuracy = accuracy(prediction, tf_labels) 

    #Will try reg later
    loss = calc_loss(fc_sent, fc_ont) 

    #Run the Adam Optimiser(AdaGrad + Momentum) with an initial eta of 0.0001
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#Number of iterations
num_steps = 12
batch_size = 100

#The error plot lists
train_errors = []
test_errors = []
time_val = []
losses = []
prev_loss = 0
initial = 0

ctr_step = 0

# Save the model
tf.get_collection('validation_nodes')
batch_sizes = [10653, 10653, 10653, 10660]
test_datasize = 7520
train_errors = []
test_errors = []
train_accuracies = []
test_accuracies = []

with tf.Session(graph=graph) as session:
  session.run(tf.global_variables_initializer())  
  
  #tf_test_labels = tf.convert_to_tensor(Y_test,np.float32)  
  #Define a variable to save graph state
  saver = tf.train.Saver()

  print("Initialized")
  # Add opts to the collection
  tf.add_to_collection('validation_nodes', tf_worddataset)
  tf.add_to_collection('validation_nodes', tf_ontdataset)
  tf.add_to_collection('validation_nodes', tf_labels)
  tf.add_to_collection('validation_nodes', prediction)
  loss_train = 0 


  for step in range(initial,num_steps):
    
    #Choose a random portion of 50 datapoints, by shuffling the data and slicing it
    #X and Y share the same first dim, need to shuffle the rows in both based on same seed, hence need to do this
    #Convert it to one epoch run - 1000 runs on a batch of 50
    #Y_shuff = Y[seed]
    
    #Over different batch files
    for batch_i in range(1,5):
        num_sgd = np.arange(batch_sizes[batch_i-1])
        np.random.shuffle(num_sgd)
        sent_bf = open("../dumps/wordRep" + str(batch_i) + "train.pkl","rb")
        sent_batchdata = pickle.load(sent_bf,encoding='latin1')
        sent_batchdata = sent_batchdata[num_sgd]
        sent_bf.close()
        ont_bf = open("../dumps/ontRep" + str(batch_i) + "train.pkl","rb")
        ont_batchdata = pickle.load(ont_bf, encoding = 'latin1')
        ont_batchdata = ont_batchdata[num_sgd]
        ont_bf.close()
        truth_bf = open("../dumps/truth" + str(batch_i) + "train.pkl","rb")
        truth_batchdata = pickle.load(truth_bf)
        truth_batchdata = truth_batchdata[num_sgd]

        truth_bf.close()
        print("Shape 0f the file data  ",sent_batchdata.shape," labels ",truth_batchdata.shape," Ont ",ont_batchdata.shape)
        i=0
        for epoch in range(batch_size,batch_sizes[batch_i-1],batch_size):
            #index1 = np.random.choice(sgd_batchsize,8,replace=False)
            train_sent_batch = sent_batchdata[i:epoch].reshape([-1, 12,250, 1])
            train_ont_batch = ont_batchdata[i:epoch].reshape([-1,50,250,1])
            train_label_batch = truth_batchdata[i:epoch].reshape([-1,1827])
            #print("Shape 0f the batch ",train_label_batch.shape," inp ",train_sent_batch.shape," Ont ",train_ont_batch.shape)
            i=epoch        
            #Assigning the batch values to keys of a feed dictionary, that will be passed around for every session run val
            #The train step runs one run of forward prop and back prop
            batch_preds, _ = session.run([prediction, train_step],feed_dict={tf_worddataset:train_sent_batch, tf_ontdataset: train_ont_batch, tf_labels: train_label_batch})

            #At every 10 steps calc loss and reset it
            if ((i/batch_size)%10 == 0):
                #Randomly choose the 300 samples
                index2 = np.random.choice(batch_sizes[batch_i -1], 600,replace=False)
                train_sent_rand = sent_batchdata[index2].reshape([-1,12,250,1])
                train_label_rand = truth_batchdata[index2].reshape([-1,1827])
                train_ont_rand = ont_batchdata[index2].reshape([-1,50,250,1])
                #Load test data
                sent_btf = open("../dumps/wordReptest.pkl","rb")
                sent_batchtdata = pickle.load(sent_btf,encoding='latin1')
                sent_btf.close()
                ont_btf = open("../dumps/ontReptest.pkl","rb")
                ont_batchtdata = pickle.load(ont_btf, encoding = 'latin1')
                ont_btf.close()
                truth_btf = open("../dumps/truthtest.pkl","rb")
                truth_batchtdata = pickle.load(truth_btf)
                truth_btf.close()

                t_index = np.random.choice(test_datasize, 400, replace=False)
                test_sents = sent_batchtdata[t_index].reshape([-1,12,250,1])
                test_labels = truth_batchtdata[t_index].reshape([-1,1827])
                test_onts = ont_batchtdata[t_index].reshape([-1,50,250,1])
                print("Shape of the test  ", test_sents.shape, " y ",test_labels.shape, " Shape of ont ",test_onts.shape)
                _,train_acc,loss_train = session.run([prediction,accuracy,loss],feed_dict={tf_worddataset:train_sent_rand, tf_ontdataset: train_ont_rand, tf_labels: train_label_rand})
                test_preds,test_acc,loss_test = session.run([prediction,accuracy,loss],feed_dict={tf_worddataset:test_sents, tf_ontdataset: test_onts, tf_labels: test_labels})

                prev_loss = loss_train

                print("\n Run ",str(i/batch_size)," within epoch ",step, " with train accuracy ", train_acc, " and test accuracy ",test_acc)
                print("Value of loss function on train ",loss_train," Loss test ", loss_test)
                train_errors.append(loss_train)
                test_errors.append(loss_test)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)

                #Print test error per class
                errors_digit = calcerror_perclass(test_preds, test_labels)
                #Store a mapping of classes against indices
                IDLabelMappingsf = open("../dumps/IDLabelMapping.json","rb")
                class_mapping = json.load(IDLabelMappingsf)
                IDLabelMappingsf.close()
                #print(" Keys ", class_mapping.keys())
                for digit in errors_digit:
                    #print(" Digit ", digit)
                    if errors_digit[digit]["val"] < 100:
                        print("Classification Error for class ",class_mapping[str(digit + 1)]," is : %.3f%%" % errors_digit[digit]["val"], " terms in class ",errors_digit[digit]["size"])

                #Clearing memory
                sent_batchtdata =[]
                truth_batchtdata = []
                ont_batchtdata = []

                #Reset loss and loss_ctr value
                ctr_step = 0
            ctr_step += 1
        print("\n Finished an epoch run ",step, " and loss ",loss_train, " within ",batch_i)
        errs = [train_errors,test_errors,train_accuracies,test_accuracies]
        filehandler2 = open("model/plotparams" + str(step) + str(batch_i) +  ".txt","wb")
        pickle.dump(errs,filehandler2,protocol=2)
        filehandler2.close()
        #Save the model every epoch(1000 iterations)
        dump_name = "model/my_model_" + str(step) + str(batch_i)
        save_path = saver.save(session, dump_name)    