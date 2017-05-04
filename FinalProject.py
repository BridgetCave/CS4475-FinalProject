import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageStat

# read in the image, create copy
# imageName = 'hair-model-4.jpg'
# imageName = 'shutterstock_159672665.jpg'
imageName = 'anderne-1.jpg'
img = cv2.imread(imageName)
imgGrayscale = cv2.imread(imageName, 0)
anotherImgGrayscale = cv2.imread(imageName, 0)

# http://alereimondo.no-ip.org/opencv/34
# https://github.com/shantnu/FaceDetect/blob/master/face_detect.py
cascPath = "haarcascade_frontalface_alt.xml"
featureCascade = cv2.CascadeClassifier(cascPath)

features = featureCascade.detectMultiScale(imgGrayscale, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100))

# # http://stackoverflow.com/questions/11064454/adobe-photoshop-style-posterization-and-opencv
# I. Creating the Posterize Filter from Photoshop
n = 8 # I decided to use 8 for the number of levels of quantization
allColors = np.arange(0, 256)
divider = np.linspace(0, 255, n+1)[1]
quantiz = np.int0(np.linspace(0, 255, n))
color_levels = np.clip(np.int0(allColors/divider), 0, n-1)
palette = quantiz[color_levels]
postImg = palette[img]
postImg = cv2.convertScaleAbs(postImg)
cv2.imwrite('posterizedImg.jpg', postImg)

# II. Create Halftone Image
# http://stackoverflow.com/questions/10572274/halftone-images-in-python
# 1. Convert to CMYK and split
rbg_image = Image.open("postImg.jpg")
cmyk_image = rbg_image.convert('CMYK')
cmyk = cmyk_image.split()


def gcr(im, percentage):
    cmyk_im = im.convert('CMYK')
    if not percentage:
        return cmyk_im
    cmyk_im = cmyk_im.split()
    cmyk = []
    for i in xrange(4):
        cmyk.append(cmyk_im[i].load())
    for x in xrange(im.size[0]):
        for y in xrange(im.size[1]):
            gray = min(cmyk[0][x,y], cmyk[1][x,y], cmyk[2][x,y]) * percentage / 100
            for i in xrange(3):
                cmyk[i][x,y] = cmyk[i][x,y] - gray
            cmyk[3][x,y] = gray
    return Image.merge('CMYK', cmyk_im)

def halftone(cmyk_image, cmyk, sample):
    cmyk = cmyk.split()
    dots = []
    angle = 0

    for channel in cmyk:
        # 2. Rotate each separated image by 0, 15, 30 and 45 degrees
        channel = channel.rotate(angle, 0, 1)
        size = channel.size[0], channel.size[1]
        half_tone = Image.new('L', size)
        draw = ImageDraw.Draw(half_tone)
        # 3. Take the half-tone of each image
        for x in xrange(0, channel.size[0], sample):
            for y in xrange(0, channel.size[1], sample):
                box = channel.crop((x, y, x + sample, y + sample))
                stat = ImageStat.Stat(box)
                diameter = (stat.mean[0] / 255) ** 0.5
                edge = 0.5 * (1 - diameter)
                x_pos, y_pos = (x + edge), (y + edge)
                box_edge = sample*diameter
                draw.ellipse((x_pos, y_pos, x_pos + box_edge, y_pos + box_edge), 255)
        # 4. Rotate back each half-toned image
        half_tone = half_tone.rotate(-angle, 0, 1)
        width_half, height_half = half_tone.size
        xx = (width_half - cmyk_image.size[0]) / 2
        yy = (height_half - cmyk_image.size[1]) / 2
        half_tone = half_tone.crop((xx, yy, xx + cmyk_image.size[0], yy + cmyk_image.size[1]))
        dots.append(half_tone)
        angle += 15
    return dots

cmyk = gcr(cmyk_image, 0)
dots = halftone(cmyk_image, cmyk, 8)
new = Image.merge('CMYK', dots)
new = new.convert('RGB')

# http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
# https://en.wikipedia.org/wiki/Canny_edge_detector
# III. Recreate the EdgeFilter on copy
edgeThresh = 1
lowThreshold = 0
max_lowThreshold = 10
ratio = 3
kernel_size = (3, 3)
window_name = "Edge Map"

length, width = imgGrayscale.shape
edgeFilterImg = cv2.GaussianBlur(imgGrayscale, (7, 7), 0)


# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
sigma = 0.33
v = np.median(edgeFilterImg)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edgeFilterImg = cv2.Canny(edgeFilterImg, lower, upper)
for i in range(0, 3):
    edgeFilterImg = cv2.dilate(edgeFilterImg, (3, 3))
inverseImg = (255 - edgeFilterImg)
inverseImgColor = cv2.cvtColor(inverseImg, cv2.COLOR_GRAY2RGB)

cv2.imwrite('inverseImgColor.jpg', inverseImgColor)
new.save('halftoneImage.bmp')

posterImg = cv2.imread('halftoneImage.bmp') * (inverseImgColor/255)

# IV. Add Border
posterImg = cv2.rectangle(posterImg, (0, 0), (posterImg.shape[1], posterImg.shape[0]), 0, 18)
posterImg = cv2.rectangle(posterImg, (0, 0), (posterImg.shape[1], posterImg.shape[0]), (255, 255, 255), 8)

# V. Add Speech Bubble using feature detection


if len(features) > 0:
    (faceX, faceY, faceW, faceH) = features[0]
    face_center = (faceX + faceW/2, faceY + faceH/2)
    imageCenterX = posterImg.shape[1] / 2
    imageCenterY = posterImg.shape[0] / 2
    ellipseX = 0
    ellipseY = 0
    if face_center[0] > imageCenterX:
        ellipseX = int(imageCenterX * 0.5)
    else:
        ellipseX = int(imageCenterX * 1.5)
    if face_center[1] > imageCenterY:
        ellipseY = int(imageCenterY * 0.5)
    else:
        ellipseY = int(imageCenterY * 1.5)
    xAxis = int(imageCenterX * 0.333)
    yAxis = int(imageCenterY * 0.333)
    poly = cv2.ellipse2Poly((ellipseX, ellipseY), (xAxis, yAxis), 0, 0, 360, 5)
    posterImg = cv2.fillConvexPoly(posterImg, poly, (255, 255, 255))
    posterImg = cv2.ellipse(posterImg, ((ellipseX, ellipseY), (xAxis * 2, yAxis * 2), 0), (0, 0, 0), 4)
    posterImg = cv2.putText(posterImg, "WOW!!", (ellipseX - 80, ellipseY + 20), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)



cv2.imwrite('FinalImage.jpg', posterImg)
