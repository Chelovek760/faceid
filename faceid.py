
from deepface.basemodels import  Facenet
from deepface.commons import functions

import numpy as np

def get_features_vec(pic:np.ndarray) -> np.ndarray:
    """
    :argument
        pic: input PIL.Image.open object
    :return
        features_vec: features vec np array
    """
    pic=np.array(pic)
    model = Facenet.loadModel()
    input_shape = functions.find_input_shape(model)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]
    img=functions.preprocess_face(pic,target_size=(input_shape_y, input_shape_x),enforce_detection = True, detector_backend = 'mtcnn')
    features_vec=model.predict(img)[0, :]
    return features_vec

def get_distance_two_features_vec(vec1:np.ndarray,vec2:np.ndarray,distance_type:str='euclidean')->np.float:
    """
        :argument
            vec1: input, output features from get_features np array
            vec1: input, output features from get_features np array
            distance_type: cos,euclidean
        :return
            distance:  distance between vec1 vec2 by distance_type
        """
    distance=np.float(0)
    if distance_type == 'cos':
        a = np.matmul(np.transpose(vec1), vec2)
        b = np.sum(np.multiply(vec1, vec1))
        c = np.sum(np.multiply(vec2, vec2))
        distance=1 - (a / (np.sqrt(b) * np.sqrt(c)))
    if distance_type == 'euclidean':
        distance = vec1 - vec2
        distance = np.sum(np.multiply(distance, distance))
        distance = np.sqrt(distance)
    return distance

