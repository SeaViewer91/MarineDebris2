import numpy as np
import cv2
import os

def SplitImg(img, SplitSize):
	height, width, _ = img.shape

	numHSplit = int(height/SplitSize)
	numWSplit = int(width/SplitSize)

	SplitImg = []

	for i in range(numHSplit):
		for j in range(numWSplit):
			SplitImg.append(img[i * SplitSize : (i+1) * SplitSize, j * SplitSize : (j+1) * SplitSize])


	return(np.stack(SplitImg))


def MergeImg(img, SplitImg, SplitSize):
	# img : 병합할 때 기준이 되는 원본 이미지
	# SplitImg : 분할 이미지를 append 한 변수 

	height, width, _ = img.shape

	numHSplit = int(height/SplitSize)
	numWSplit = int(width/SplitSize)

	total = []
	p = 0

	for i in range(numHSplit):
		row = []
		for j in range(numWSplit):
			row.append(SplitImg[p])
			p = p + 1
		total.append(np.concatenate(row, axis=1))

	return(np.concatenate(total, axis=0))