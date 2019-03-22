from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from PIL import Image, ImageFilter
import pickle
import requests
from io import BytesIO
import base64
import pprint

# Create your views here.
from django.http import HttpResponse
from django.template import loader


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    # creates white canvas of 28x28 pixels
    newImage = Image.new('L', (28, 28), (255))

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        # resize height according to ratio width
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
        # calculate horizontal position
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        # resize width according to ratio height
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
        # caculate vertical pozition
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva


def index(request):
    template = loader.get_template('numberOracle/index.html')
    context = {
        'NumberOracle': ['This page is numberOracle.'],
    }
    return HttpResponse(template.render(context, request))


def ajaxform(request):
    template = loader.get_template('numberOracle/test_form.html')
    context = {'test_message_list': [
        'hello, there?', 'my name is Hong.', 'How are you, today?'], }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def searchData(request):
    msg = request.POST.getlist('msg[]')

    # data = request.POST.get('msg', '')
    # print(data)
    # s1 = base64.b64decode(data[1])
    # print(s1)
    pixels = []
    a = []
    # for i in range(0, 784):
    #    pixels.append(msg[i])
    print(len(msg))
    #a = msg.split(',')
    # print(a)
    print(len(a))
    for i in range(0, len(msg)):
        pixels.append(int(msg[i])/255)
    print(pixels)
    # for i in range(0, 28*28):
    #    pixels.append(msg[i])
    # print(pixels[0])

    # pixels = [
    #    [(54, 54, 54), (232, 23, 93), (71, 71, 71), (168, 167, 167)],
    #    [(204, 82, 122), (54, 54, 54), (168, 167, 167), (232, 23, 93)],
    #    [(71, 71, 71), (168, 167, 167), (54, 54, 54), (204, 82, 122)],
    #    [(168, 167, 167), (204, 82, 122), (232, 23, 93), (54, 54, 54)]
    # ]
    # with open("imageToSave.png", "wb") as fh:
    #    fh.write(base64.decodebytes(data))
    # Convert the pixels into an array using numpy
    # array = np.array(pixels, dtype=np.uint8)
    # pprint.pprint(pixels)
    # Use PIL to create an image from the new array of pixels
    # new_image = Image.fromarray(array)
    # new_image.save('new.png')

    # x = imageprepare('new.png')  # file path here
    # print(len(x))  # mnist IMAGES are 28x28=784 pixels
    # Load model from file
    pprint.pprint(np.array(pixels).reshape(1, 28, 28, 1))

    with open("mnist_model.pkl", 'rb') as file:
        mnist_model = pickle.load(file)
    A = mnist_model.predict_classes(
        np.array(pixels).reshape((1, 28, 28, 1)))
    print('The Answer is ', str(A[0]))
    context = {'msg': str(A[0]), }

    file.close()
    return HttpResponse(json.dumps(context), "application/json")
