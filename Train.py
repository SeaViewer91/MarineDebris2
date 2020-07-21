from imageai.Detection.Custom import DetectionModelTrainer

# WeightPATH = 'data/models/' + 'detection_model-ex-030--loss-0005.590.h5'
# WeightPATH = 'YOLOv3(2020-06-10).h5'
WeightPATH = 'pretrained-yolov3.h5'

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="data")
trainer.setTrainConfig(
	object_names_array=["Styrofoam", "PET", "Plastic", "NoTarget"], 
	batch_size=6, 
	num_experiments=50, 
	train_from_pretrained_model=WeightPATH)
trainer.trainModel()