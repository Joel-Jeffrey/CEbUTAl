import csv
import os
import pydicom
from PIL import Image
import cv2
import os, shutil
import numpy as np

# Path to DICOM images and location to save processed images
imgfolder = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT/UCSD/UCSD Data/AuthorFold/Data'
text_folder = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT/UCSD/UCSD Data/AuthorFold/F1train.csv'
Train1 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/train/1'
Train0 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/train/0'
Test1 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/test/1'
Test0 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/test/0'
Val1 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/val/1'
Val0 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/OCT_Binary/val/0'

def emptydir(dir_path):
	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)
		os.makedirs(dir_path)
	else:
		os.makedirs(dir_path)

Dirlist = [Train0, Train1, Test0, Test1, Val1, Val0]
for i in range(len(Dirlist)):
	emptydir(Dirlist[i])

Label = ["CNV","DME","DRUSEN","NORMAL"]
CNV_path = os.path.join(imgfolder,Label[0])
DME_path = os.path.join(imgfolder,Label[1])
DRUSEN_path = os.path.join(imgfolder,Label[2])
NORMAL_path = os.path.join(imgfolder,Label[3])

# Counters to facilitate splitting
c_0 = 0
c_1 = 0
c_2 = 0
c_3 = 0

# Store path for train, val and test data
# All variables with index 0 -> CNV, index 1 -> DME, index 2 -> DRUSEN, index 3 -> NORMAL
l_0 = []
l_1 = []
l_2 = []
l_3 = []
t_0 = []
t_1 = []
t_2 = []
t_3 = []
v_0 = []
v_1 = []
v_2 = []
v_3 = []
i = 0

# The train-val-test split numbers 
train0 = 7500
train1 = 2500
train_tot0 = train0 + train1
train2 = 5000
train3 = 35000
train_tot1 = train2 + train3
val0 = 1000
val1 = 1000
val_tot0 = val0 +val1
val2 = 1000
val3 = 1000
val_tot1 = val3 +val2
test0 = 1000
test1 = 1000
test_tot0 = test0 + test1 
test3 = 250
test2 = 250
test_tot1 = test3 + test2

# Read and store path in the lists defined above
with open(text_folder, 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter='_')
	for row in csv_reader:
		if(i==0):
			i+=1
			continue
		r = row[0].split(',')
		if(len(l_0)==train0 and len(l_1)==train1 and len(l_2)==train2 and len(l_3)==train3 and len(v_0)==val0 and len(v_1)==val1 and len(v_2)==val2 and len(v_3)==val3 and len(t_0)==test0 and len(t_1)==test1 and len(t_2)==test2 and len(t_3)==test3):
			break	
		if(r[1]=='0'):
			if(c_0<train0):
				l_0.append(str(r[0]))
				c_0+=1
			else:
				if (c_0-train0)<val0:
					v_0.append(str(r[0]))
					c_0+=1
				else:
					if (c_0-train0-val0)<test0:
						t_0.append(str(r[0]))
						c_0+=1
		if(r[1]=='1'):
			if(c_1<train1):
				l_1.append(str(r[0]))
				c_1+=1
			else:
				if (c_1-train1)<val1:
					v_1.append(str(r[0]))
					c_1+=1
				else:
					if (c_1-train1-val1)<test1:
						t_1.append(str(r[0]))
						c_1+=1
		if(r[1]=='2'):
			if(c_2<train2):
				l_2.append(str(r[0]))
				c_2+=1
			else:
				if (c_2-train2)<val2:
					v_2.append(str(r[0]))
					c_2+=1
				else:
					if (c_2-train2-val2)<test2:
						t_2.append(str(r[0]))
						c_2+=1
		if(r[1]=='3'):
			if(c_3<train3):
				l_3.append(str(r[0]))
				c_3+=1
			else:
				if (c_3-train3)<val3:
					v_3.append(str(r[0]))
					c_3+=1
				else:
					if (c_3-train3-val3)<test3:
						t_3.append(str(r[0]))
						c_3+=1		
					

# print(len(l_0),len(l_1),len(l_2),len(l_3),len(v_0),len(v_1),len(v_2),len(v_3),len(t_0),len(t_1),len(t_2),len(t_3))
# print(c_0,c_1,c_2,c_3)
# print(len(l_0)+len(l_1),len(l_3)+len(l_2),len(v_0)+len(v_1),len(v_3)+len(v_2),len(t_0)+len(t_1),len(t_3)+len(t_2))

# Perform normalization, convert to png and save
for i in range(train0):
	file_path = os.path.join(CNV_path, l_0[i])
	fname = l_0[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Train1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(train1):
	file_path = os.path.join(DME_path, l_1[i])
	fname = l_1[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Train1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(train2):
	file_path = os.path.join(DRUSEN_path, l_2[i])
	fname = l_2[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Train0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(train3):
	file_path = os.path.join(NORMAL_path, l_3[i])
	fname = l_3[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Train0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')
  
for i in range(val0):
	file_path = os.path.join(CNV_path, v_0[i])
	fname = v_0[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Val1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(val1):
	file_path = os.path.join(DME_path, v_1[i])
	fname = v_1[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Val1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(val2):
	file_path = os.path.join(DRUSEN_path, v_2[i])
	fname = v_2[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Val0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(val3):
	file_path = os.path.join(NORMAL_path, v_3[i])
	fname = v_3[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Val0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(test0):
	file_path = os.path.join(CNV_path, t_0[i])
	fname = t_0[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Test1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(test1):
	file_path = os.path.join(DME_path, t_1[i])
	fname = t_1[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Test1, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(test2):
	file_path = os.path.join(DRUSEN_path, t_2[i])
	fname = t_2[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Test0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

for i in range(test3):
	file_path = os.path.join(NORMAL_path, t_3[i])
	fname = t_3[i].split('.')[0]
	with Image.open(file_path) as img:
		resized_image = img.resize((512, 512))
		full_path = os.path.join(Test0, f"{fname}.png")
		resized_image.save(full_path, format='PNG')

print("Saved successfully\n\n")
