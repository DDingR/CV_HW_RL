
import cv2
import numpy as np

def extract(state):

    state = np.uint8(
        np.array(
            state * 255.0
        )
    )
    img_list = []
    # print(state.shape)
    for i in range(int(state.shape[2]/3)):
        img_list.append(
            np.reshape(state[:,:,3*i:3*i+3], [state.shape[0], state.shape[0], 3])
        )
        # print(state.shape)
        # print(state[:,:,i:i+3].shape)
    img_list = np.array(img_list)

    # cv2.imwrite('./img/tmp1.png', img_list[0])
    # cv2.imwrite('./img/tmp2.png', img_list[1])
    # cv2.imwrite('./img/tmp3.png', img_list[2])
    # cv2.imwrite('./img/tmp4.png', img_list[3])
    
    result = []

    for img in img_list:

        # 흑백 영상으로 변환(원을 검출하기 위해)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 허프 변환 함수가 노이즈에 민감하기 때문에 가우시안 블러로 노이즈 제거
        blr = cv2.GaussianBlur(gray, (0, 0), 1)

        # 허프 변환 원 검출
        # 트랙바를 이용한 테스트로 파라미터 값 결정
        circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=25, param2=25/2, minRadius=15, maxRadius=50)


        tmp = posCheck(circles)
        result.append(
            tmp
        )
        
    result = np.reshape(
        result,
        [1,40]
    )
    return result

def posCheck(circles):

    tmp = np.zeros([1,10])

    if circles is None:
        return tmp

    for circle in circles[0]:
        x, y, r = circle

        if r > 30:
            tmp[0, 8:10] = [x, y]
        
        if x > 250:
            if y > 250:
                tmp[0, 0:2] = [x, y]
            else:
                tmp[0, 2:4] = [x, y]
        if x < 250:
            if y < 250:
                tmp[0, 4:6] = [x, y]
            else:
                tmp[0, 6:8] = [x, y]

    return tmp

def visualCheck(state):

    ori_state = state.copy()

    # pre process
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.GaussianBlur(state, (0, 0), 1)

    cv2.namedWindow("Trackbar Test")

    cv2.createTrackbar("minDistance", "Trackbar Test", 1, 255, lambda x : x)
    cv2.createTrackbar("param1", "Trackbar Test", 1, 255, lambda x : x)
    # cv2.createTrackbar("param2", "Trackbar Test", 1, 255, lambda x : x)
    cv2.createTrackbar("minRadius", "Trackbar Test", 1, 255, lambda x : x)
    cv2.createTrackbar("maxRadius", "Trackbar Test", 1, 255, lambda x : x)

    cv2.setTrackbarPos("minDistance", "Trackbar Test", 50)
    cv2.setTrackbarPos("param1", "Trackbar Test", 25)
    # cv2.setTrackbarPos("param2", "Trackbar Test", 20)
    cv2.setTrackbarPos("minRadius", "Trackbar Test", 15)
    cv2.setTrackbarPos("maxRadius", "Trackbar Test", 50)


    while cv2.waitKey(1) is not ord('q'):
        tmp = ori_state.copy()

        minDistance = cv2.getTrackbarPos("minDistance", "Trackbar Test")
        param1 = cv2.getTrackbarPos("param1", "Trackbar Test")
        # param2 = cv2.getTrackbarPos("param2", "Trackbar Test")
        minRadius = cv2.getTrackbarPos("minRadius", "Trackbar Test")
        maxRadius = cv2.getTrackbarPos("maxRadius", "Trackbar Test")

        cv2.putText(
            tmp,
            "minDis: " + str(minDistance) + " param1: " + str(param1) + " min r: " + str(minRadius) + " max r: " + str(maxRadius),
            (50,100),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,0,0), 2
        )

        circles = cv2.HoughCircles(state, cv2.HOUGH_GRADIENT, 1, minDistance,
                                    param1=param1, param2=param1/2, minRadius=minRadius, maxRadius=maxRadius)

        print(circles)
        if circles is not None:
            print(circles.shape)
            for circle in circles[0]:
                x, y, radius = circle

                print(x, y)

                cv2.circle(
                    tmp,
                    (int(x), int(y)), int(radius),
                    (0, 0, 255),
                    2, cv2.LINE_AA
                )

        cv2.imshow("Tracbar Test", tmp)          
