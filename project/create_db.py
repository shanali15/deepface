import warnings

import cv2
def analysis():
	
	input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
	
	text_color = (255,255,255)
	
	employees = []
	#check passed db folder exists
	if os.path.isdir(db_path) == True:
		for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
			for file in f:
				if ('.jpg' in file):
					#exact_path = os.path.join(r, file)
					exact_path = r + "/" + file
					#print(exact_path)
					employees.append(exact_path)
					
	if len(employees) == 0:
		print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")
	
	#------------------------
	
	if len(employees) > 0:
		
		model = DeepFace.build_model(model_name)
		print(model_name," is built")
		
		#------------------------
		
		input_shape = functions.find_input_shape(model)
		input_shape_x = input_shape[0]
		input_shape_y = input_shape[1]
		
		#tuned thresholds for model and metric pair
		threshold = dst.findThreshold(model_name, distance_metric)
		
	#------------------------
	#facial attribute analysis models
		
	if enable_face_analysis == True:
		pass
	#------------------------
	
	#find embeddings for employee list
	
	tic = time.time()
	
	pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
	
	embeddings = []
	#for employee in employees:
	for index in pbar:
		employee = employees[index]
		pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
		embedding = []
		img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False)
		img_representation = model.predict(img)[0,:]
		
		embedding.append(employee)
		embedding.append(img_representation)
		embeddings.append(embedding)
	
	df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
	df['distance_metric'] = distance_metric
	
	toc = time.time()
	
	print("Embeddings found for given data set in ", toc-tic," seconds")
	
	#-----------------------

	pivot_img_size = 112 #face recognition result image

	#-----------------------
	
	opencv_path = functions.get_opencv_path()
	face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(face_detector_path)
	
	#-----------------------

	freeze = False
	face_detected = False
	face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
	freezed_frame = 0
	tic = time.time()
	cap = cv2.VideoCapture(source) #webcam
	while(True):
		ret, img = cap.read()
		raw_img = img.copy()
		resolution = img.shape
		
		resolution_x = img.shape[1]; resolution_y = img.shape[0]

		if freeze == False: 
			faces = face_cascade.detectMultiScale(img, 1.3, 5)
			
			if len(faces) == 0:
				face_included_frames = 0
		else: 
			faces = []
		
		detected_faces = []
		face_index = 0
		for (x,y,w,h) in faces:
			if w > 130: #discard small detected faces
				
				face_detected = True
				if face_index == 0:
					face_included_frames = face_included_frames + 1 #increase frame for a single face
				
				cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
				
				cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
				
				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
				
				#-------------------------------------
				
				detected_faces.append((x,y,w,h))
				face_index = face_index + 1
				# print(face_index)
				#-------------------------------------
				
		if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			freeze = True
			#base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()
		
		if freeze == True:

			toc = time.time()
			if (toc - tic) < time_threshold:
				
				if freezed_frame == 0:
					freeze_img = base_img.copy()
					#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue
					
					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]
												
						cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
						
						#-------------------------------
						
						#apply deep learning for custom_face
						
						custom_face = base_img[y:y+h, x:x+w]
						
						#-------------------------------
						#facial attribute analysis
						
						if enable_face_analysis == True:
							pass

						custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False)
						
						#check preprocess_face function handled
						if custom_face.shape[1:3] == input_shape:
							if df.shape[0] > 0: #if there are images to verify, apply face recognition
								img1_representation = model.predict(custom_face)[0,:]
								
								#print(freezed_frame," - ",img1_representation[0:5])
								
								def findDistance(row):
									distance_metric = row['distance_metric']
									img2_representation = row['embedding']
									
									distance = 1000 #initialize very large value
									if distance_metric == 'cosine':
										distance = dst.findCosineDistance(img1_representation, img2_representation)
									elif distance_metric == 'euclidean':
										distance = dst.findEuclideanDistance(img1_representation, img2_representation)
									elif distance_metric == 'euclidean_l2':
										distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
										
									return distance
								
								df['distance'] = df.apply(findDistance, axis = 1)
								df = df.sort_values(by = ["distance"])
								
								candidate = df.iloc[0]
								employee_name = candidate['employee']
								best_distance = candidate['distance']
								
								print(candidate[['employee', 'distance']].values)
								
								#if True:
								if best_distance <= threshold:
									#print(employee_name)
									display_img = cv2.imread(employee_name)
									
									display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
																		
									label = employee_name.split("/")[-1].replace(".jpg", "")
									label = re.sub('[0-9]', '', label)
									print(label)
									date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
									val = label,date+".jpg",1,date
									print(label,type(label))
									sql = "INSERT INTO UserActivities ([User],Image,Activity,AddedDate) VALUES (?,?,?,?)"
									cursor.execute(sql, val)
									cnxn.commit()
									print(label)
						tic = time.time() #in this way, freezed image can show 5 seconds
						
						#-------------------------------
				
				time_left = int(time_threshold - (toc - tic) + 1)
				
				cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
				cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
				
				cv2.imshow('img', freeze_img)
				
				freezed_frame = freezed_frame + 1
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0
			
		else:
			cv2.imshow('img',img)
		
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break
		
	#kill open cv things		
	cap.release()
	cv2.destroyAllWindows()

analysis(source=0)