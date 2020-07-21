from xml.etree.ElementTree import Element, dump, ElementTree

def indent(elem, level=0):
	i = '\n' + level*'\t'
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + '\t'
		if not elem.tail or not elem.tail.strip():
			elem.tail = i

		for elem in elem:
			indent(elem, level+1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i

class annot:
	def __init__(self, FolderName, ImgName, Detection, ExportPath):
		self.FolderName		= FolderName
		self.ImgName		= ImgName
		self.Detection		= Detection
		self.ExportPath		= ExportPath

		self.root 			= self.AnnotationRoot(self.FolderName, self.ImgName)

		for temp in self.Detection:

			self.root = self.AnnotationObject(self.root, 
											temp['name'],
											temp['boxinfo'][0], temp['boxinfo'][1], temp['boxinfo'][2], temp['boxinfo'][3])

		indent(self.root)
		ElementTree(self.root).write(self.ExportPath + '/' + self.ImgName[:-4] + '.xml')


	def ObjectMaker(self, ObjectName, XMin, YMin, XMax, YMax):

		o = Element('object')

		objectName = Element('name')
		objectName.text = ObjectName
		o.append(objectName)
	
		objectPose = Element('pose')
		objectPose.text = 'Unspecified'
		o.append(objectPose)

		objectTruncated = Element('truncated')
		objectTruncated.text = '0'
		o.append(objectTruncated)

		objectDifficult = Element('Difficult')
		objectDifficult.text = '0'
		o.append(objectDifficult)

		objectBndbox = Element('bndbox')
		xmin = Element('xmin')
		xmin.text = str(XMin)
		objectBndbox.append(xmin)

		ymin = Element('ymin')
		ymin.text = str(YMin)
		objectBndbox.append(ymin)

		xmax = Element('xmax')
		xmax.text = str(XMax)
		objectBndbox.append(xmax)

		ymax = Element('ymax')
		ymax.text = str(YMax)
		objectBndbox.append(ymax)
		o.append(objectBndbox)

		return o

	def AnnotationRoot(self, FolderName, ImgName):
		root = Element('annotation')
		folder = Element('folder')
		folder.text = self.FolderName
		root.append(folder)

		filename = Element('filename')
		filename.text = self.ImgName
		root.append(filename)

		path = Element('path')
		path.text = FolderName + '/' + ImgName
		root.append(path)

		source = Element('source')
		database = Element('database')
		database.text = 'Unknown'
		source.append(database)

		size = Element('size')
		width = Element('width')
		width.text = '608'
		size.append(width)
		height = Element('height')
		height.text = '608'
		size.append(height)
		depth = Element('depth')
		depth.text = '3'
		size.append(depth)
		root.append(size)

		segmented = Element('segmented')
		segmented.text = '0'
		root.append(segmented)

		return root


	def AnnotationObject(self, root, ObjectName, xmin, ymin, xmax, ymax):

		Object = self.ObjectMaker(ObjectName, xmin, ymin, xmax, ymax)
		root.append(Object)

		return root


if __name__ == '__main__':

	Detection = list()
	temp = dict()
	temp['name'] = 'Styrofoam'
	temp['boxinfo'] = (10, 20, 15, 19)

	Detection.append(temp)
	print(Detection)

	temp = dict()
	temp['name'] = 'PET'
	temp['boxinfo'] = (25, 17, 30, 41)

	Detection.append(temp)

	print(Detection)

	T = annot(FolderName = 'TestFolder', 
				ImgName = 'TestImg.jpg', 
				Detection = Detection,
				ExportPath = 'export')