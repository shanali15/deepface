from deepface import DeepFace
obj = DeepFace.analyze(img_path = "test\download.jpg", actions = ['age', 'gender', 'race', 'emotion'])
#objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])