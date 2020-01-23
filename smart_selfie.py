import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(r"C:\Users\hp\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier(r"F:\Users\hp\Desktop\mini_project_6th\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(r"C:\Users\hp\Anaconda3\Lib\site-packages\cv2\data\haarcascade_smile.xml")
    
cap = cv2.VideoCapture(0)
    
while True:
    ret, img = cap.read()
    if ret is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        continue
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    temp=0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
        #eyes = eye_cascade.detectMultiScale(roi_gray)
    
        smile = smile_cascade.detectMultiScale(roi_gray,
                                                   scaleFactor=1.82,
                                                   minNeighbors=22,
                                                   minSize=(50,50),
                                                   )
    
        #for (ex, ey, ew, eh) in eyes:
         #   cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        temp=0
        for (sx, sy, sw, sh) in smile:
            temp=sx
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,255), 2)
    if temp:
        img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        num_down = 2 # number of downsampling steps
        num_bilateral = 7 # number of bilateral filtering steps


        # downsample image using Gaussian pyramid
        img_color = img1
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        # upsample image to original size
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        #STEP 2 & 3
        #Use median filter to reduce noise
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        #STEP 4
        #Use adaptive thresholding to create an edge mask
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=9,
                                         C=2)

        # Step 5
        # Combine color image with edge mask & display picture
        # convert back to color, bit-AND with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)

        # display
        
        plt.imshow(img_cartoon)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        break
    else:
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()