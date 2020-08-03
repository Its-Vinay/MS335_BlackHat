import numpy as np
import tensorflow as tf
import cv2 as cv
import time
import os

def uniform(*args):
	# Read the graph.

	directory = r'C:\\Users\\Pritika\\Desktop\\police\\police\\static\\uni_outputs'

	with tf.io.gfile.GFile('C:\\Users\\Pritika\\Desktop\\police\\Uniform_detection\\frozen_inference_graph.pb', 'rb') as f:
		graph_def = tf.compat.v1.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.compat.v1.Session() as sess:
	    # Restore session
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')

	    # Read and preprocess an image.
		if(args[0]==0):
			cap = cv.VideoCapture(0)
			check,img = cap.read()
			print('Wait for 3 seconds...')
			time.sleep(3)
		else:
			img=cv.imread(args[0])

		rows = img.shape[0]
		cols = img.shape[1]
		inp = cv.resize(img, (300, 300))
		inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

	    # Run the model
		out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
					sess.graph.get_tensor_by_name('detection_scores:0'),
					sess.graph.get_tensor_by_name('detection_boxes:0'),
					sess.graph.get_tensor_by_name('detection_classes:0')],
					feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
		class_names=['belt','cap','dori','nameplate']
	    
		a=[]
	    # Visualize detected bounding boxes.
		num_detections = int(out[0][0])
		for i in range(num_detections):
			classId = int(out[3][0][i])
			score = float(out[1][0][i])
			bbox = [float(v) for v in out[2][0][i]]
			if score > 0.3:
				label = class_names[classId-1]
				a.append(classId)

				x = bbox[1] * cols
				y = bbox[0] * rows
				right = bbox[3] * cols
				bottom = bbox[2] * rows
				img = cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
				cv.putText(img,label, (int(x), int(y-10)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (36,255,12), 2)
		b=[]
		sum=0
		string=""
		for j in range(1,5):
			for i in a:
				if(j==i):
					sum+=1
					break
			if(sum==0):
				b.append(j)

		if(len(b)==0):
			print('uniform is OK')
			string="Nothing"
		else:
			for k in b:
				print(class_names[k-1]+",",end=" ")
				string=string+class_names[k-1]+" "
			print("are/is missing") 
		


	cv.imshow('Result',img)
	os.chdir(directory)
	filename='uniform_output.jpg'
	cv.imwrite(filename, img)
	cv.waitKey(5000)
	cv.destroyAllWindows()
	return("missing - "+ string)