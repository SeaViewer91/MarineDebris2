from PIL import Image
from PIL.ExifTags import TAGS

def get_GPSInfo(img):
    
    # Get EXIF Info
    info = img._getexif()
    
    # Decode EXIFInfo
    exif = {}
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif[decoded] = value
    # Read GPSInfo
    exifGPS = exif['GPSInfo']
    latData = exifGPS[2]
    lonData = exifGPS[4]
    
    # calculate the lat / long
    latDeg = latData[0][0] / float(latData[0][1])
    latMin = latData[1][0] / float(latData[1][1])
    latSec = latData[2][0] / float(latData[2][1])
    lonDeg = lonData[0][0] / float(lonData[0][1])
    lonMin = lonData[1][0] / float(lonData[1][1])
    lonSec = lonData[2][0] / float(lonData[2][1])
    
    # Degree/Min/Sec to Degree
    Lat = latDeg + latMin/60 + latSec/3600
    Lon = lonDeg + lonMin/60 + lonSec/3600

    return(Lat, Lon)


