import glob

def to_cla(category):
	if category == "Car":
		return 0
	elif category == "Van":
		return 1
	elif category == "Truck":
		return 2
	elif category == "Pedestrian":
		return 3
	elif category == "Person_sitting":
		return 4
	elif category == "Cyclist":
		return 5
	elif category == "Tram":
		return 6
	elif category == "Misc":
		return 7
	else:
		return -1

def to_bbox(box):
	return [box["x1"], box["y1"], box["x2"], box["y2"]]

label = glob.glob("/home/liu/Desktop/KITTI/training/label_2/*.txt")
images = glob.glob("/home/liu/Desktop/KITTI/data_object_image_2/training/image_2/*.png")
print (len(label), len(images))

kitty_train = open("kitti_train.txt",'w') 
for l in label:
	label_file = open(l,'r') 
	data = label_file.readlines()

	info = "/home/liu/Desktop/KITTI/data_object_image_2/training/image_2/" + l.split("/")[-1].split(".")[0] + ".png "
	
	objs = []
	for line in data:
		elts = line.split(" ")
		cla = to_cla(elts[0])
		if cla == -1:
			continue
		bbox = [elts[4], elts[5], elts[6], elts[7]]
		objs.append([cla, bbox, elts[3], elts[8:11]])

	if len(objs) < 1:
		continue

	for obj in objs:
		for value in obj[1]:
			info = info + str(int(float(value))) + ","
		info = info + str(int(obj[0])) + "," + obj[2] + ","
		for value in obj[3]:
			info = info + value + ","
		info = info[:-1] + " "
		
	info = info[:-1] + "\n"
	kitty_train.write(info)
kitty_train.close()
