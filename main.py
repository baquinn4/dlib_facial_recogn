import dlib
import cv2
import os
import time
from multiprocessing import Process
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Queue
import sys
from math import sqrt
import imutils
from imutils import face_utils
import shutil
import random
import traceback

#UNKNOWN_HAS_FACE_DIR = "Unknown_has_face"

# "have face_parse" traverses through dir of random photos and finds ones with faces
# for some reason hog is really accurate in some photos, detecting all faces and in others half or < half
#-- i think the due to size(#pixels) difference 

# 1) gets all image names from target directory
# 2) cd to target directory
# 3) for each image in target directory use hog to detect face, then draw rectangle around it
# 4) save all images with faces to a list of images
# 5) cd ../
# 6) save all images to new directory with new names, all as same format
# note: if image has no face we do nothing with it, we dont want it 
# 8/11/21 : going to try and add multi-processing, spreading out dataset over all cores to reduce 
#           time taken

# file paths to downloaded pretrained dlib models
SHAPE_PREDICTOR_PRETRAINED = "/home/brad/Documents/projects/P-Hark/FindFace/shape_predictor_5_face_landmarks.dat"
FACE_REC_PRETRAINED_MODEL = "/home/brad/Documents/projects/P-Hark/FindFace/dlib_face_recognition_resnet_model_v1.dat"

target_dir_name = "Pictures_to_use"
names = []
chunked_data = []

for name in os.listdir(target_dir_name):
    names.append(name)

# split our list of picture names into n sized chunks
def get_chunked_data():
    return chunked_data

# names: list of all photo names in target_dir
# n: how many N-sized lists to chunk names into
def chunk(names,n):
    for i in range(0,len(names),n):
        yield names[i: i + n]

# q: queue passed via Process object
# pnum: process number


faces_hog = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(SHAPE_PREDICTOR_PRETRAINED)
facerec = dlib.face_recognition_model_v1(FACE_REC_PRETRAINED_MODEL)


# this class contains the 128D vector describing each face found
# is a c++ vector object so i may need cython


def euclidean_distance(face_vta,face2_vta):
    return sqrt(sum((x1-x2)**2 for x1,x2 in zip(face_vta,face2_vta)))

# if euclidean distance < 0.6 then face descriptors are generally the same person
def compare_face(a,b):
    dist = euclidean_distance(a,b)
    return dist

#parses photos to see if has face or not
def hf_parse(pnum,q,face_objects_q):
    
    names = q.get()
    count = 0
    passed_images = []
    failed_images = []
    face_objects = {}
    
    for n in names:

        #print(f"READING FROM FILE: {n}")
        image = cv2.imread(n)
        if image is None:
            continue
        
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            faces = faces_hog(image, 0)
            if len(faces) != 0:
                for face in faces:

                    shape = sp(image,face)

                    fd = facerec.compute_face_descriptor(image,shape) #,100,0.3
                    fd = np.array(fd)
                    face_objects[n + "_" + str(count)] = fd
                    (x,y,w,h) = face_utils.rect_to_bb(face)
                    cv2.rectangle(image, (abs(x), abs(y)), (abs(x) + abs(w), abs(y) + abs(h)), (0, 0, 255), 2)
                    face_image = image[y:y+h,x:x+w]
                    
                    if os.path.isdir("Face_Pictures"):
                        os.chdir("Face_Pictures")
                        cv2.imwrite(f"{n}_{count}.bmp",face_image)
                        os.chdir("../")
                    else:
                        os.mkdir("Face_Pictures")
                        os.chdir("Face_Pictures")
                        cv2.imwrite(f"{n}_{count}.bmp",face_image)
                        os.chdir("../")
                    
                    count+=1
                passed_images.append(image)
            else:
                failed_images.append(image)

        except (TypeError,cv2.error) as ce:
                #if ce == TypeError:
                    ##print(f"Photo: {n} throwing !_img.empty()")
                #else:
                    #print(f"Process: {pnum} Photo: {n} \n{str(ce)}")
                pass
                count+=1
                  
                        
    os.chdir("../")
    
    os.chdir("Passed_Pictures")

    countp = 0
    for i in passed_images:
        cv2.imwrite(str(pnum) + ":"  + str(countp) + ".jpg" ,i)
        countp +=1
    
    os.chdir("../")
    os.chdir("Failed_Pictures")

    countf = 0
    for i in failed_images:
        cv2.imwrite(str(pnum) + ":"  + str(countf) + ".jpg" ,i)
        countf +=1

    face_objects_q.put(face_objects)

    #print(len(face_objects))
    

if __name__=="__main__":

    start = time.time()

    print("Running")

    num_processes = cpu_count()
    processes_IDs = list(range(0,num_processes))

    processes = []

    num_images_per_process = len(names)/float(num_processes)
    num_images_per_process = int(np.ceil(num_images_per_process))
   
    chunked_data = list(chunk(names,num_images_per_process))

    os.chdir(target_dir_name)
    
    q = Queue()

    face_objects_q = Queue()
    
    for i in range(num_processes):
        q.put(chunked_data[i])
        process = Process(target=hf_parse,args=(i,q,face_objects_q))
        processes.append(process)

    for process in processes:
        process.start()

    face_objects = []
    for process in processes:
        fo = face_objects_q.get()
        face_objects.append(fo)

    for process in processes:
        process.join()
   

    # just loosely written code for testing to get face descriptors and test comparison
    #for i in range(0,len(face_objects)):
        #for key,value in face_objects[i].items():
            #if key == "IMG_0288.JPG_24":
                #print(f" ONE: {i}")
            #elif key == "IMG_0291.PNG_46":
                #print(f"TWO: {i}")
            #else:
                #pass
            #print(f"Photo name: {key} Face Descriptor: {value}")
    #print(len(face_objects))

    # face object dictionary from list of face object dictionarys: face_objects
    #fod_1 = face_objects[0]
    #fod_2 = face_objects[3]

    # getting value : face descriptor (128 dimension vector) from key : photo_name
    #face_vta_1 = fod_1['IMG_0288.JPG_24']
    #face_vta_2 = fod_2['IMG_0291.PNG_46']
    #dist = compare_face(face_vta_1,face_vta_2)
    #print(dist)

    # could probably optimize this and make it faster
    # each element of face_objects is a dict (key: photo name,value: face descriptor 128D vector)
    # so merging all 4 dictionaries from the 4 individual processes
    # then extracting all values into an array of face_descriptor numpy arrays for comparison


        

    #key: IMG_0143.PNG_49 : + ".bmp" will give us filename of face 
    #key: IMG_0291.PNG_45 : + ".bmp" will give us filename of face

    os.chdir("../")

    #print(f"Num distinct_faces: {distinct_faces}")
    end = time.time()
    print(f"Runtime: {end-start}")

    

    