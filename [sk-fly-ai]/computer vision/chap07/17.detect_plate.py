import numpy as np, cv2

while True:
    no = int(input("차량 영상 번호( 0:종료 ) : "))             # 차량 번호 입력
    if no == 0: break;

    fname = "chap07/images/test_car/{0:02d}.jpg".format(no)
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    if image is None:
        print(str(no) + "번 영상 파일이 없습니다.")
        continue
    
    mask = np.ones((5, 17), np.uint8)                      # 닫힘 연산 마스크
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 명암도 영상 변환
    gray = cv2.blur(gray, (5, 5))                           # 블러링
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 5)              # 소벨 에지 검출
    # gray = cv2.Canny(gray, 100,200)  # 소벨 에지 검출

    # 이진화 및 닫힘 연산 수행
    _, th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, mask, iterations=3)

    cv2.imshow("image", image)
    cv2.imshow("binary image", th_img)
    cv2.imshow("opening", morph)
    cv2.waitKey()
