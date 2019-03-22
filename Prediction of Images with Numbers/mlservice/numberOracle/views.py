from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here
from django.http import HttpResponse
from django.template import loader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import pickle
import cv2

np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 12

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load model from file
model_filename = "numberOracle/mnist_model.pkl"
with open(model_filename, 'rb') as file:
    mnist_model = pickle.load(file)

n = 30
print('The Answer is ', mnist_model.predict_classes(x_test[n].reshape((1, 28, 28, 1))))



img = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0.,  0.23921569, 0.6117647,  0.5803922,  0.34901962, 0.31764707, 0.6117647,  1.,         0.99607843, 0.99607843, 0.99607843, 0.84705883, 0.6117647, 0.5803922,  0.16078432, 0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.10196079,
  0.8117647,  0.99607843, 0.99215686, 0.99215686, 0.99215686, 0.9607843,
  0.91764706, 0.7647059,  0.9137255,  0.9137255,  0.9137255,  0.9647059,
  0.99215686, 0.94509804, 0.09411765, 0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.6509804,
  0.99215686, 0.99607843, 0.99215686, 0.99215686, 0.5019608,  0.1882353,
  0.,         0.,         0.,         0.,         0.,         0.1882353,
  0.8862745,  0.99215686, 0.34117648, 0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.01568628, 0.68235296, 0.9843137,
  0.99215686, 0.95686275, 0.4862745,  0.07450981, 0.00784314, 0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.8392157,  0.99215686, 0.07450981, 0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.5176471,  0.99215686, 0.99215686,
  0.69411767, 0.1764706,  0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.18039216,
  0.95686275, 0.99215686, 0.07450981, 0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.5019608,  0.7647059,  0.3529412,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.7019608,
  0.99607843, 0.59607846, 0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.03137255, 0.99607843,
  0.9843137,  0.28235295, 0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.64705884, 0.99607843,
  0.64705884, 0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.14509805, 0.972549,   0.99607843,
  0.18039216, 0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.3019608,  0.99215686, 0.67058825,
  0.03529412, 0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.07450981, 0.9372549,  0.99607843, 0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.00784314, 0.36078432, 0.99215686, 0.99215686, 0.84313726,
  0.8392157,  0.8392157,  0.70980394, 0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.07450981, 0.654902,   0.99215686, 0.99215686, 0.99215686, 0.99607843,
  0.99215686, 0.99215686, 0.8352941,  0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.7882353,  0.99215686, 0.99215686, 0.99215686, 0.7254902,  0.45882353,
  0.13333334, 0.07450981, 0.0627451,  0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.46666667, 0.28627452, 0.75686276, 0.92941177, 0.14117648, 0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.3137255,  0.99607843, 0.8392157,  0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.4627451,  0.99215686, 0.65882355, 0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.4627451,  0.99215686, 0.45882353, 0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.4627451,  0.99215686, 0.45882353, 0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.28235295, 0.75686276, 0.13333334, 0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.        ],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]

print('The answer is ', mnist_model.predict_classes(np.reshape(img, (1,28,28,1))))


def index(request):
    template = loader.get_template('numberOracle/index.html')
    context = {
        'index_test': ['Index Testing Page by YMK', 'Is it working?'],
    }
    return HttpResponse(template.render(context, request))

def ajaxform(request):
    template = loader.get_template('numberOracle/test_form.html')
    context = {'test_message_list': ['hello, there?', 'my name is YMK.', 'How are you, today?'],}
    return HttpResponse(template.render(context,request))

@csrf_exempt
def searchData(request):
    data = request.POST['msg']
    context = {'msg': data,}
    return HttpResponse(json.dumps(context), "application/json")

@csrf_exempt
def oracleNumber(request):
    # y_pre = [0,1,2,3,4,5,6,7,8,9,0,3,2,5,7,2,9,6,2,7,3,7,9,3,0,6,8,4,9,0,6,3,5,7,8,3,5,4,2,4,5,9,3,0,6,8,2,0,1,1,5,8,6,4,8,9,2,9,0,6,8,4,9,6,0,2,9,5,7,0,
    # 5,0,9,3,0,2,8,5,9,3,8,9,6,5,4,3,2,5,9,0,3,4,7,9,4,8,2,8,6,2]
    y_pre=[0,1,2,3,4,5,6,7,8,9,4,9,2,9,4,9,5,9,5,8,3,1,0,8,8,5,9,5,3,2,1,0,5,7,7,3,7,2,5,2,0,6,9,3,2,4,7,0,2,5,7,4,3,0,1,7,3,0,5,9,7,4,5,8,0,3,6,5,8,4,6,3,2,9,8,7,2,0,8,3,1,2,0,2,8,5,0,5,7,3,2,3,0,5,9,4,8,3,2,0]
    data = request.POST['msg']
    #data2 = json.loads(data)
    #context = {'msg':  data,}
    print(data)
    data2 = json.loads(data)
    print(data2)
    # print('type(data)=', type(data), ',  ', 'type(data2)=', type(data2), ',  ', 'type(data2[0])=', type(data2[0]))
    img_ori = cv2.imread("./numberOracle/static/human.jpg",cv2.IMREAD_GRAYSCALE)
    img1 = img_ori.copy()
    ret, thresh = cv2.threshold(img_ori,127,255,0)
    _, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rects = []
    print(thresh.shape)
    im_w = thresh.shape[1]
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 30 and h < 32 :
            print("too small size: ", w, h)
            continue
        if w > im_w / 5:
            print("too largh size: ", w, h)
            continue
        y2 = round(y / 40) * 40
        index = y2 * im_w + x
        rects.append((index, x, y, w, h))
        img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imwrite('test_img_predict.png', img1)
    rects = sorted(rects, key=lambda x : x[0])

    X = []
    for i, r in enumerate(rects):
        index, x, y, w, h = r
        num = thresh[y-10:y+h+10, x-10:x+w+10]
        num = np.array(num)
        num = cv2.bitwise_not(num)
        num = cv2.resize(num,(28,28))
        num = num.reshape(1, 28,28,1)
        num = num.astype('float32')
        n = mnist_model.predict_classes(num)
        X.append(n[0])

    print('number of digits found', len(rects))
    for i,item in enumerate(X):
        print(item,end=' ')
        if i % 10==9:
            print()

    result = ""
    acc = 0
    for i in range(100):
        if y_pre[i] == X[i]:
            acc += 1
        result += str(y_pre[i]) + " "
        if i % 10 == 9:
            result += "<br />"

    result += "accuracy: %f"%(acc)+"%"
    print("accuracy: %f"%(acc),"%")



    context = {'msg': str(result),}
    return HttpResponse(json.dumps(context), "application/json")
