print('Mode (1) : Monitoring, (2) : Test')
mode = int(input('Mode : '))

import time

stime = time.time()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

from PIL import Image
from utils.ImgSM import SplitImg, MergeImg
from utils.exif import get_GPSInfo
from utils.xmlMaker	import annot
from lib.Detection.Custom import ObjectDetection

ImgFolderName = 'SC4'
# Create Img List
if mode == 1:

	year 	= str(datetime.today().year)
	month 	= str(datetime.today().month)
	day 	= str(datetime.today().day)
	today 	= '(' + year + '-' + month + '-' + day + ')' 

	UAVImgPath 		= 'image_' + ImgFolderName + '/'
	ExportImgPath 	= 'Output_' + ImgFolderName + today + '/'
	ReportPath 		= 'Report_' + ImgFolderName + today + '/'

	# Project에 필요한 folder들 생성
	if os.path.isdir(UAVImgPath) == False:
		print('Input Image 경로를 다시 설정하시오.')

	
	os.makedirs(ExportImgPath, exist_ok=True)
	os.makedirs(os.path.join(ExportImgPath,'window'), exist_ok=True)		
	os.makedirs(ReportPath, exist_ok=True)


elif mode == 2:
	# for Test
	UAVImgPath 		= 'test_input_img/'
	ExportImgPath 	= 'test_output_img/'


UAVImgList = os.listdir(UAVImgPath)
UAVImgList = [Img for Img in UAVImgList if Img.endswith('.jpg')]

# Load Pre-trained Model
# H5Path = 'data/models/' + 'detection_model-ex-009--loss-0005.424.h5'
H5Path = 'YOLOv3(2020-06-16).h5'
jsonPath = 'data/json/detection_config.json'

detector = ObjectDetection()
detector.YOLOv3()
detector.PretrainedModelPath(H5Path)
detector.MetadataPath(jsonPath)
detector.loadModel()

# Run Program 
'''Data Table for save counting result
Column 0 : lat
Column 1 : lon
Column 2 : Total Styrofoam
Column 3 : Total PET
Column 4 : Total Plastic(etc) '''
DataTable = np.zeros((len(UAVImgList), 5))

TotalStyrofoam 		= 0
TotalPET 			= 0
TotalPlastic 		= 0

for idx, Img in enumerate(UAVImgList):

	'''원본 이미지를 불러와서 각 이미지를 608x608 크기로 분할'''
	# 'UAVImgPath + Img' 를 불러와서 RawImg에 할당  
	# RawImg 는 Object Detection에 활용되는 이미지 
	RawImg = cv2.imread(UAVImgPath + Img, cv2.IMREAD_COLOR)

	print('Image %d Loaded!' % idx)
	print('')

	# TempImg는 EXIF에서 GPS Info를 불러오기 위한 이미지 
	TempImg = Image.open(UAVImgPath + Img)
	# get_GPSInfo()를 통해 이미지의 Lat, Lon을 구함 
	ImgLat, ImgLon = get_GPSInfo(TempImg)

	print('Image GPS Information Loaded')
	print('Lat : %f   Lon : %f' % (ImgLat, ImgLon))

	DataTable[idx, 0] = ImgLat
	DataTable[idx, 1] = ImgLon

	# ImgTotalItem : 이미지 내에서 각 Item의 수를 Count하기 위한 목적 
	# RawImg를 Window Img로 분할하여 각 Window Img에서 Object Detection을 수행한 다음 그 결과를 합산 
	# RawImg 단위로 Counting을 수행하므로 RawImg가 바뀔 때 마다 0으로 초기화 
	ImgTotalStyrofoam 	= 0
	ImgTotalPET 		= 0
	ImgTotalPlastic 	= 0

	# ExportImgName : Object Detection을 수행하여 Bounding Box가 그려진 ExportImg의 File Name
	ExportImgName = ExportImgPath + 'Detection_' + Img

	# BGR to RGB
	# RawImg는 OpenCV를 통해 불러오기 때문에 BGR Color로 되어 있음 
	# Object Detection Model은 RGB Color Image를 Input Image로 받아들이기 때문에 Color Space 변환이 필요함 
	# RawImg = cv2.cvtColor(RawImg, cv2.COLOR_BGR2RGB)

	''' Raw Img의 H 또는 W가 608보다 크거나 작은 경우 Resize'''
	# 다른 해상도를 가진 이미지를 불러올 때를 대비하여 작성한 코드 
	# 본 프로그램은 DJI Mavic 2 Pro(20MP)로 촬영된 Image를 Defualt로 함 
	RawImg_H, RawImg_W, _ = RawImg.shape

	if (RawImg_H % 608 != 0) or (RawImg_W % 608 != 0):

		RawImg = cv2.resize(RawImg,
							dsize=(int(np.ceil(RawImg_H/608)*608), int(np.ceil(RawImg_W/608)*608)),
							interpolation=cv2.INTER_AREA)

	''' Split Image
	Result of Function : Tensor
	shape = (n of Piece, 608, 608, 3)
	'''

	# RawImg를 608 * 608 * 3 Image로 분할 
	# SplitImg(InputImg, WindowSize) : InputImg를 WindowSize로 분할 
	# Output Shape : num of WindowImg, WindowSize, WindowSize, 3 
	sImg = SplitImg(RawImg, 608)

	# Detection
	# Object Detection결과를 담기 위해 dImg 생성 
	# Object Detection 후 dImg 내 WindowImg와 치환 
	dImg = sImg

	for idx2 in range(sImg.shape[0]): # sImg.shape[0] : num of WindowImg

		'''UAV 이미지를 window 크기로 나누어 각 window단위로 detection 수행'''

		# idx2번째 WindowImg를 InputImg로 할당 
		InputImg = sImg[idx2, :, :, :]
		# InputImg = np.array(InputImg).reshape(608, 608, 3)

		# Object Detection 수행 
		Detection = detector.ImageDetector(input_type='array', input_image=InputImg, output_type='info', minimum_confidence=30)

		if len(Detection) != 0:

			ImgName = Img[:-4] + '_' + str(idx2)
			# WinImg = cv2.cvtColor(InputImg, cv2.COLOR_RGB2BGR)
			WinImg = InputImg

			RawWinImg = 0
			for dResult in Detection:
				if dResult['name'] == 'Styrofoam' or dResult['name'] == 'PET' or dResult['name'] == 'Plastic':
					RawWinImg = 1

			if RawWinImg == 1:
				# Raw Images
				cv2.imwrite(ExportImgPath + 'window' + '/' + ImgName + '.jpg', InputImg)

				# Annotation
				XML = annot(FolderName = UAVImgPath,
							ImgName = ImgName + '.jpg',
							Detection = Detection,
							ExportPath = ExportImgPath + 'window' + '/')

			# Run Detection
			nTarget	= 0
			for dResult in Detection:

				if dResult['name'] == 'Styrofoam':
					nTarget 			= nTarget + 1
					ImgTotalStyrofoam 	= ImgTotalStyrofoam + 1
					TotalStyrofoam 		= TotalStyrofoam + 1

					tl = (dResult['boxinfo'][0], dResult['boxinfo'][3])
					br = (dResult['boxinfo'][2], dResult['boxinfo'][1])

					OutputWinImg = cv2.rectangle(InputImg, tl, br, color=(0, 255, 0), thickness=5)


				elif dResult['name'] == 'PET':
					nTarget 			= nTarget + 1 
					ImgTotalPET 		= ImgTotalPET + 1
					TotalPET 			= TotalPET + 1

					tl = (dResult['boxinfo'][0], dResult['boxinfo'][3])
					br = (dResult['boxinfo'][2], dResult['boxinfo'][1])

					OutputWinImg = cv2.rectangle(InputImg, tl, br, color=(255, 0, 0), thickness=5)

				elif dResult['name'] == 'Plastic':
					nTarget 			= nTarget + 1
					ImgTotalPlastic 	= ImgTotalPlastic + 1
					TotalPlastic 		= TotalPlastic + 1

					tl = (dResult['boxinfo'][0], dResult['boxinfo'][3])
					br = (dResult['boxinfo'][2], dResult['boxinfo'][1])

					OutputWinImg = cv2.rectangle(InputImg, tl, br, color=(0, 0, 255), thickness=5)

			# Print Window Images
			# dResult 내 탐지 결과가 모두 'NoTarget'인 경우 원본 이미지를 그대로 출력 

			if nTarget > 0:
				
				# OutputWinImg = cv2.cvtColor(OutputWinImg, cv2.COLOR_RGB2BGR)
				# Bounding Box Img
				cv2.imwrite(ExportImgPath + 'window' + '/' + ImgName + '(result).jpg', OutputWinImg)

				dImg[idx2, :, :, :] = OutputWinImg

			elif nTarget == 0:

				# OutputWinImg = cv2.cvtColor(InputImg, cv2.COLOR_RGB2BGR)
				OutputWinImg = InputImg
				dImg[idx2, :, :, :] = OutputWinImg			

		else:
			# OutputWinImg = cv2.cvtColor(InputImg, cv2.COLOR_RGB2BGR)
			OutputWinImg = InputImg
			dImg[idx2, :, :, :] = OutputWinImg


	# Merge
	DataTable[idx, 2] = ImgTotalStyrofoam
	DataTable[idx, 3] = ImgTotalPET
	DataTable[idx, 4] = ImgTotalPlastic


	print('')
	print('[Styrofoam : %d]  [PET : %d]  [Plastic : %d]' % (ImgTotalStyrofoam, ImgTotalPET, ImgTotalPlastic))
	print('')

	'''개별 window 탐지 결과를 병합 영상으로 변환'''
	ResultImg = MergeImg(RawImg, dImg, 608)

	cv2.imwrite(ExportImgName, ResultImg)
	print('Finish : %d/%d  [Styrofoam : %d] [PET : %d] [Plastic : %d]' % (idx+1, len(UAVImgList), TotalStyrofoam, TotalPET, TotalPlastic))


	print('')
	print(DataTable[idx, :])


print('=============================================')
print('Total Styrofoam : %d, Total PET : %d, Total Plastic : %d' % (TotalStyrofoam, TotalPET, TotalPlastic))

DataTable_filename = os.path.join(ReportPath,'Report.csv')
np.savetxt(DataTable_filename, 
			DataTable,
			fmt='%3.5f',
			delimiter=',',
			header='lat,lon,Styrofoam,PET,Plastic',
			comments='')

etime = time.time()

delta = etime - stime
print('경과 시간 : %d 초' %delta)