import csv
import os
import pydicom
from PIL import Image
import cv2
import os, shutil
import numpy as np

# Path to DICOM images and location to save processed images
dcm_folder= '/home/shared/rsna-intracranial-hemorrage-detection/rsna-intracranial-hemorrage-detection/stage_2_train'
text_file = '/home/shared/rsna-intracranial-hemorrage-detection/rsna-intracranial-hemorrage-detection/stage_2_train.csv'
Train1= '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/train/1'
Train0= '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/train/0'
Test1 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/test/test1'
Test0 = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/test/test0'
Val1= '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/val/1'
Val0= '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/ICH_Binary/val/0'

def emptydir(dir_path):
	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)
		os.makedirs(dir_path)
	else:
		os.makedirs(dir_path)

Dirlist = [Train0, Train1, Test0, Test1, Val1, Val0]
for i in range(len(Dirlist)):
	emptydir(Dirlist[i])

# The train-val-test split numbers 
train0 = 47505
train1 = 255
val0 = 2005
val1 = 2005
test0 = 205
test1 = 805

# Counters to facilitate splitting
c_0 = 0
c_1 = 0
c_2 = 0
c_3 = 0
c_4 = 0
c_5 = 0

# Store path for train, val and test data
l_0 = []
l_1 = []
t_0 = []
t_1 = []
v_0 = []
v_1 = []

# Read and store path in the lists defined above
i = 0
with open(text_file, 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter='_')
	for row in csv_reader:
		if(i==0):
			i+=1
			continue
		r = row[2].split(',')
		if(r[0]=='any'):
			if(len(l_0)==train0 and len(l_1)==train1 and len(v_0)==val0 and len(v_1)==val1 and len(t_0)==test0 and len(t_1)==test1):
				break
			if(r[1]=='0'):
				if(c_0<train0):
					l_0.append(str(row[1]))
					c_0+=1
				else:
					if c_2<val0:
						v_0.append(str(row[1]))
						c_2+=1
					else:
						if c_4<test0:
							t_0.append(str(row[1]))
							c_4+=1
			if(r[1]=='1'):
				if(c_1<train1):
					l_1.append(str(row[1]))
					c_1+=1
				else:
					if c_3<val1:
						v_1.append(str(row[1]))
						c_3+=1
					else:
						if c_5<test1:
							t_1.append(str(row[1]))
							c_5+=1

# print(len(l_0),len(l_1),len(v_0),len(v_1),len(t_0),len(t_1))
# print(c_0,c_1,c_2,c_3,c_4,c_5)

l_0 = list((f"ID_{i}.dcm" for i in l_0))
l_1 = list((f"ID_{i}.dcm" for i in l_1))
v_0 = list((f"ID_{i}.dcm" for i in v_0))
v_1 = list((f"ID_{i}.dcm" for i in v_1))
t_0 = list((f"ID_{i}.dcm" for i in t_0))
t_1 = list((f"ID_{i}.dcm" for i in t_1))

# Perform normalization, convert to png and save
for i in range(train0):
	dcm_file_path = os.path.join(dcm_folder, l_0[i])
	fname = l_0[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Train0, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)

for i in range(train1):
	dcm_file_path = os.path.join(dcm_folder, l_1[i])
	fname = l_1[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Train1, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)
  
for i in range(val0):
	dcm_file_path = os.path.join(dcm_folder, v_0[i])
	fname = v_0[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Val0, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)

for i in range(val1):
	dcm_file_path = os.path.join(dcm_folder, v_1[i])
	fname = v_1[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Val1, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)

for i in range(test0):
	dcm_file_path = os.path.join(dcm_folder, t_0[i])
	fname = t_0[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Test0, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)

for i in range(test1):
	dcm_file_path = os.path.join(dcm_folder, t_1[i])
	fname = t_1[i].split('.')[0]
	ds = pydicom.dcmread(dcm_file_path)
	new_image = ds.pixel_array.astype(float)
	scaled_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255
	scaled_image = np.uint8(new_image)
	final_image = Image.fromarray(scaled_image)
	resized_image = final_image.resize((512, 512))
	full_path = os.path.join(Test1, f"{fname}.png")
	# print(f"Saving image to: {full_path}")
	resized_image.save(full_path)

print("Saved successfully\n\n")
