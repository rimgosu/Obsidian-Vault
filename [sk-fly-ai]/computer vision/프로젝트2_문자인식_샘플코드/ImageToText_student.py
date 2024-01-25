import cv2
import numpy as np
import pytesseract

TESSERACT_PATH = "C:/Program Files/Tesseract-OCR/tesseract.exe" #테서렉스 설치 경로
imgpath='./imgs/2.jpg'  #이미지 파일 경로
win_name = "Image To Text"  #OpenCV 창 이름
img = cv2.imread(imgpath)   #이미지 읽어오기



#마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):

    return 0


#이미치 처리 함수
def ImgProcessing():
    
    return 0


#OCR 함수
def GetOCR():
    #이미지 불러오기
    global img

    #OCR모델 불러오기
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    #OCR모델로 글자 추출
    text = pytesseract.image_to_string(img, lang='kor+eng')
        
    return text


cv2.imshow(win_name, img)   #이미지 출력
cv2.waitKey(0)              #입력 대기
text = GetOCR()             #OCR함수로 텍스트 추출
print(text)                 #텍스트 출력