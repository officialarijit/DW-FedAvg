
import numpy as np
import random
#import cv2
import os
#from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K



# def load(paths, verbose=-1):
#     '''expects images for each class in seperate dir, 
#     e.g all digits in 0 class in the directory named 0 '''
#     data = list()
#     labels = list()
#     # loop over the input images
#     for (i, imgpath) in enumerate(paths):
#         # load the image and extract the class labels
#         im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
#         image = np.array(im_gray).flatten()
#         label = imgpath.split(os.path.sep)[-2]
#         # scale the image to [0, 1] and add to list
#         data.append(image/255)
#         labels.append(label)
#         # show an update every `verbose` images
#         if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
#             print("[INFO] processed {}/{}".format(i + 1, len(paths)))
#     # return a tuple of the data and labels
#     return data, labels


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}



def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dense(50))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))
        return model
    

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = bce(Y_test, logits)
    Y_prdt=np.where(logits > 0.5, 1, 0)
    Y_test  = Y_test.numpy() #converting Y_true from Tenosr to Numpy
    acc = accuracy_score(Y_test, Y_prdt)
    F1=f1_score(Y_test, Y_prdt, zero_division=1)
    precision=precision_score(Y_test, Y_prdt, zero_division=1)
    recall=recall_score(Y_test, Y_prdt, zero_division=1)
    
    
    cm = confusion_matrix(Y_test, Y_prdt)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, logits)
    auc_value = roc_auc_score(Y_test, logits)
    
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {} | global_f1: {} | global_precision: {} | global_recall: {} | global_auc: {}| flobal_FPR: {} '.format(comm_round, acc, loss, F1, precision, recall, auc_value, FPR))
    return acc, loss, F1, precision, recall, auc_value, FPR

