import json

def to_cla(category):
	if category == "bike":
		return 0
	elif category == "bus":
		return 1
	elif category == "car":
		return 2
	elif category == "motor":
		return 3
	elif category == "person":
		return 4
	elif category == "rider":
		return 5
	elif category == "traffic light":
		return 6
	elif category == "traffic sign":
		return 7
	elif category == "train":
		return 8
	elif category == "truck":
		return 9
	else:
		return -1

def to_bbox(box):
	return [box["x1"], box["y1"], box["x2"], box["y2"]]

with open('../bdd100k/labels/bdd100k_labels_images_train.json') as f:
    data = json.load(f)

print len(data)

bdd_train = open("bdd_train.txt",'w') 
for label in data:
	info = "/home/liu/Desktop/YOLOv3-tensorflow/bdd100k/images/100k/train/" + label['name'] + " "
	objs = []
	for obj in label['labels']:
		cla = to_cla(obj['category'])
		if cla == -1:
			continue
		bbox = to_bbox(obj['box2d'])
		objs.append([cla, bbox])

	if len(objs) < 1:
		continue

	for obj in objs:
		for value in obj[1]:
			info = info + str(int(value)) + ","
		info = info + str(int(obj[0])) + " "
		
	info = info[:-1] + "\n"
	bdd_train.write(info)
bdd_train.close()
