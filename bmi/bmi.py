import joblib
import numpy as np
import face_recognition
import cv2

def bmi(*args):
	bmi_model = joblib.load('bmi_predictor.model')

	def get_face_encoding(image_path):
		if(args[0]==0):
			my_face_encoding = face_recognition.face_encodings(image_path)
		else:
			picture_of_me = face_recognition.load_image_file(image_path)
			my_face_encoding = face_recognition.face_encodings(picture_of_me)
		if not my_face_encoding:
			print("no face found !!!")
			return np.zeros(128).tolist()
		return my_face_encoding[0].tolist()

	def predict_height_width_BMI(test_image,bmi_model):
		test_array = np.expand_dims(np.array(get_face_encoding(test_image)),axis=0)
		bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
		return bmi
	if(args[0]==0):
		s,sum1=0,0
		video = cv2.VideoCapture(0)
		while True:
			check,frame = video.read()
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			cv2.imshow('Image',gray)
			a = predict_height_width_BMI(frame,bmi_model)
			sum1+=a		#Total_BMI
			s+=1			#Total iteration
			key = cv2.waitKey(1)
			if key==ord('q'):
				break
		print("BMI : ",sum1/s)
		video.release()
		cv2.destroyAllWindows()
	else:
		a = predict_height_width_BMI(args[0],bmi_model)
		print(a)

if __name__=="__main__":
	bmi('index.jpeg')


