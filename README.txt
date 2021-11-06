Description
--------------

hf_parse(directory)

"have face_parse" traverses through dir of random photos and finds ones with faces.

# 1) gets all image names from target directory.
# 2) cd to target directory.
# 3) for each image in target directory use hog to detect face, then draw  rectangle around it.
# 4) save all images with faces to a list of images.
# 5) cd ../
# 6) save all images to new directory, Parsed_pictures dir, with new names, 	    all as same format.
# note: if image has no face we do nothing with it, until the end
# 7) saves failed pics(no faces) to failed_pictures dir.
