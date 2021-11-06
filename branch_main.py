import dlib
import cv2
import os
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import cpu_count


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



# ---parameters----
# td : string, target_dir, a directory of photos to be parsed for faces
# names : array, photo filenames from target_dir 
# n : int, number of N-sized chunks to generate for multiprocessing
# container: list, will contain the lists of image objs

import dlib
import cv2
import os
import time
from multiprocessing import Process
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count

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


list_of_images = []
list_of_fails = []
target_dir_name = "test"
names = []

for name in os.listdir(target_dir_name):
    names.append(name)

# split our list of picture names into n sized chunks

def chunk(names,n):
    for i in range(0,len(names),n):
        yield names[i: i + n]

#list of picture names from chunked data
def hf_parse(names):

    count = 0

    os.chdir(target_dir_name)

    print(f"Changing to:{target_dir_name}")

    for n in names:
    
        print("reading file:" + n)

        image = cv2.imread(n)

        fn = str(count) + "_img"
        
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_hog = dlib.get_frontal_face_detector()
        faces = faces_hog(image, 0)
        #dets,scores,idx = faces_hog.run(image,1,-1)
        #for i,d in enumerate(dets):
        #print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))

        if len(faces) != 0:
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
           
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            print(f"Adding: {n} : to list of images")
                #cv2.imshow("Image", image)
                #cv2.waitKey(1000)
                #time.sleep(1)
            #list_of_images.append(image)
        #else:
            #list_of_fails.append(image)


        #end = time.time()
        #print(f"Runtime: {end-start} ")



def save_pictures():

    os.chdir("../")
    os.chdir("Parsed_Pictures")
    #print("Moving to: Parsed_Pictures")
    #print(f"\nLength of images list is: {str(len(list_of_images))}")

    countp = 0
    for i in list_of_images:
        print(f"Saving image: {str(countp)}")
        cv2.imwrite("new_" + str(countp) + ".jpg" ,i)
        countp += 1

    os.chdir("../")
    os.chdir("Failed_Pictures")

    countf = 0
    for i in list_of_fails:
        cv2.imwrite("failed_" + str(countf) + ".jpg",i)
        countf += 1






if __name__=="__main__":

    start = time.time()

      
    hf_parse(names)
    #save_pictures()

    

    end = time.time()
    print(f"Runtime: {end-start} ")
    
#runtime: 10.9766 
#this code does not use multiprocessing