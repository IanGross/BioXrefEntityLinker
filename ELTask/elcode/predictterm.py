import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import matplotlib.lines as mlines


with tf.Session() as sess:
    # Load old model
    new_saver = tf.train.import_meta_graph('model/my_modelplain_38.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('model/'))

    #Helper function to access the tensors by names
    def T(layer):
        '''Helper for getting layer output tensor'''
        return tf.get_default_graph().get_tensor_by_name("{}:0".format(layer))

    '''
    Find per class Classification Error
    Find the argmax and compare the indices of the argmax at a per class level
    Returns a dictionary of the losses against digits
    This will later be leveraged to find Classification errors for each class
    '''
    def calcerror_perclass(predictions,labels,class_mapping):
        digits = {i:{"val":0,"size":0} for i in range(0,1827)}
        print("Shape of predictions ", predictions.shape)
        #Need to apply a softmax during the testing step
        preds_arg = predictions
        labels_np = labels
        #Converting both of them to np.array, so that it is easy loop and store values
        preds_labels = np.argmax(preds_arg,1)
        labels_indices = np.argmax(labels_np,1)
        preds_size = len(preds_labels)
        print(" Pred size ", preds_size)

        wordf = open('../dumps/word_contextstestSet.pkl','rb')
        word_contexts = pickle.load(wordf)
        wordf.close()

        print("Size of mapping ",len(class_mapping))

        print("Shape of preds labels",preds_labels.shape,"Shape of labels",labels_indices.shape,"Size of actual array",preds_size)
        for pred_index in range(0,preds_size):
            print("Entity : ", word_contexts[pred_index]["entity"], " with neighborhood ",  word_contexts[pred_index]["consolidated"], " Ground truth ", word_contexts[pred_index]["truth"], " Predicted ",class_mapping[str(preds_labels[pred_index] + 1)] )
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


    '''
    Calculate the mean squared loss
    Calc loss for every data point, and take the mean
    '''
    def calc_loss(prediction,labels):
        #Averaging out the loss
        return tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=prediction))

    #Defining the placeholders
    X = T('tf_dataset')
    X_ont = T('tf_ontdataset')
    Y = T('tf_labels')

    #Retrieving values and operations from from the restored graph

    prediction = T('pred_op')
    loss = calc_loss(prediction,Y)

    X_senttest=pickle.load(open("../dumps/wordReptestset.pkl","rb"),encoding='latin1')

    print("Shape of X is ", X_senttest.shape)

    X_onttest=pickle.load(open("../dumps/ontReptestset.pkl","rb"),encoding='latin1')

    print("Shape of X ont is ", X_onttest.shape)
   
    Y_test=pickle.load(open("../dumps/truthtestset.pkl","rb"),encoding='latin1')
    print("Shape of Y  is ",Y_test.shape)

    X_senttest = X_senttest.reshape([-1, 12,250, 1])
    X_onttest = X_onttest.reshape([-1,50,250,1])
    Y_test = Y_test.reshape([-1,1827])

    predicted_y = sess.run(prediction, {X: X_senttest, Y: Y_test, X_ont: X_onttest})

    
    #Store a mapping of classes against indices
    IDLabelMappingsf = open("../dumps/IDLabelMapping.json","rb")
    class_mapping = json.load(IDLabelMappingsf)
    IDLabelMappingsf.close()
    #Print test error per class
    errors_digit = calcerror_perclass(predicted_y, Y_test, class_mapping)
    #print(" Keys ", class_mapping.keys())
    for digit in errors_digit:
        #print(" Digit ", digit)
        if errors_digit[digit]["val"] < 100:
            print("Classification Accuracy for class ",class_mapping[str(digit + 1)]," is : %.3f%%" % (100 - errors_digit[digit]["val"]), " terms in class ",errors_digit[digit]["size"])

    print("Shape of predictions ",predicted_y.shape, " Actual ", Y_test.shape)    