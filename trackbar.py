import cv2
import numpy as np

def extract(state):

    # unin8
    state = np.uint8(
        np.array(
            state * 255.0
        )
    )

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

if __name__=="__main__":
    img = cv2.imread("tmp3.png")
    extract(img)