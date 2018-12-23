from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input
from keras.applications.densenet import decode_predictions
from keras.applications.densenet import DenseNet169

model = DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
for i in range(0, 5):
    print('-----------------------------------------------')
    # load an image from file
    image = load_img('../resources/wolf_'+str(i+1)+'.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    for i in range(0, len(label[0])):
        # retrieve the most likely result, e.g. highest probability
        labelTemp = label[0][i]
        # print the classification
        print('%s (%.2f%%)\n' % (labelTemp[1], labelTemp[2]*100))
