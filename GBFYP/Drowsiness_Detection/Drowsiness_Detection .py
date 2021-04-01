from scipy.spatial import distance
from imutils import face_utils
import imutils
import cv2
import dlib
import playsound
from playsound import playsound
#import matplotlib.pyplot as plt
#from multiprocessing import Process, Queue
#import time
#import pickle

#def set_alarm():
#	min= 1

#	t=datetime.now()
#	sec=min*10
#exec(open('detector.py').read())
#eye espect ratio  (X-Axis, Y-Axis)
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


#function to plot graph
#def threadedPlotter(ear, itera):

	# print("HEY")
	# time.sleep(5)
	# RUNNING = False
	#print(ear)
	#ax = plt.subplot(111)
	#plt.plot(itera, ear)
	# plt.savefig("plot.png")
	#pickle.dump(ax, open('plot.pickle', 'wb'))	


thresh = 0.25           #Threshold on which the counter will start counting frames

frame_check2= 5         #Number of Frames on which it will show warning
ALARM_ON = False

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FPS, 20) #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da7c2fa550ba270713fca1405397b90ae0
### setting up points for feeding matplotlib
#count = 1
#iteration = []
exit = False
while True:
	ret, frame=cap.read()                      
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	ear = None
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check2:
				cv2.putText(frame, "***************ALERT !!!!! ****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************WAKE UP***************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
				
				playsound('/home/pi/Desktop/Drowsiness_Detection/alert-sound.mp3')
				#cap.release()
				#cv2.destroyAllWindows()
				exit = True
				break
			#ear_val.append(ear)
			#iteration.append(count)
			#count += 1
		else:
			ALARM_ON = False
			flag = 0
		
		
		#threadedPlotter(ear_val, iteration)
	#if exit: break
	cv2.imshow("Drowsiness Detecetion", frame)
	key = cv2.waitKey(100) & 0xFF == ord("q")
      #break
	#if key == ord("q"):
		# print(iteration)
		# plt.plot(iteration, ear_val)
		# plt.show()

cv2.destroyAllWindows()
cap.stop()
		# plotter.stop()
              
	
	
