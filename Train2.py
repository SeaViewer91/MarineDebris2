from imageai.Detection.Custom import DetectionModelTrainer

# WeightPATH = 'data/models/' + 'detection_model-ex-009--loss-0005.424.h5'
# WeightPATH = 'pretrained-yolov3.h5'
WeightPATH = 'pretrained-resnet.h5'

trainer = DetectionModelTrainer()
trainer.setModelTypeAsRetinaNet()
trainer.setDataDirectory(data_directory="data")
trainer.setTrainConfig(
	object_names_array=["Styrofoam", "PET", "Plastic", "NoTarget"], 
	batch_size=8, 
	num_experiments=50, 
	train_from_pretrained_model=WeightPATH)
trainer.trainModel()