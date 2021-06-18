import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--mtcnn-model', default='../model/mtcnn-model/', help='path to load mtcnn model.')
parser.add_argument('--model', default='../model/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--use-gpu', default=False, type=bool, help='use gpu')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('Tom_Hanks_54745.png')
print(img.shape)

# import json
# img = cv2.resize(img, (224, 224))
# print(img.shape)
# img_dict_str = '{"data": %s}' % str(img.tolist())
# data = json.loads(img_dict_str)
# img = np.asarray(data['data'], dtype=np.uint8)
# print(img.shape)
    
img = model.get_input(img)
f1 = model.get_feature(img)
print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
sys.exit(0)
img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
