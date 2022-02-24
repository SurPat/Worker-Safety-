from personCounting_yolov4 import detect_people_1
from scipy.spatial import distance as dist
import numpy as np
import cv2;
import glob
import time;
import psutil
import os


# from firebase import firebase

def cam1(number):
    # # Define the max/min safe distance limits (in pixels) between 2 people.
    MAX_DISTANCE = 170  # 50 #80
    MIN_DISTANCE = 70  # 25 #50
    # # Set the threshold value for total violations limit.
    Threshold = 15

    # load yolo
    # net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    global vidpath
    global fname

    classes = []
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # prin(classes)
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # camera open
    cap = cv2.VideoCapture(0)

    global fname
    global vidpath
    #      cap = cv2.VideoCapture(0)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(FPS)
    ########################
    out_vid = cv2.VideoWriter("cam{0}_result.avi".format(number), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS,
                              (int(cap.get(3)), int(cap.get(4))))  ########################
    ########################
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["classes.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    while True:
        _, frame = cap.read()
        frame_id += 1
        print('frame_id:', frame_id)
        # height, width, channels = frame.shape
        try:
            results = detect_people_1(frame, net, ln=output_layers, personIdx=LABELS.index("person"))
            serious = set()
            abnormal = set()
            areaArr = []
            bbAreaArr = []
            if len(results) >= 2:
                print("Results:", results)
                # extract all centroids from the results and compute the
                #                 # Euclidean distances between all pairs of the centroids
                # centroids = np.array([r[2] for r in results])
                # print("centroids:", centroids)
                coordinates = np.array([c[1] for c in results])
                coordinates = coordinates.tolist()
                centroid = []
                print("Count of Received coordinates:", len(coordinates))
                print("results: ", results)
                for a in results:
                    print("coordinate:", a)
                    area = ((a[1][2] - a[1][0]) * (a[1][3] - a[1][1])) / (25.4 * 25.4)
                    print("Area:", area)
                    if area < 45 or area > 90:
                        print("removing coordinate ", a)
                        results.remove(a)
                for a in results:
                    print("coordinate:", a)
                    area = ((a[1][2] - a[1][0]) * (a[1][3] - a[1][1])) / (25.4 * 25.4)
                    bbAreaArr.append(area)
                print("results:", results)
                centroids = np.array([r[2] for r in results])
                # bbAreaArr.append([a(3) for a in coordinates])
                print("Count of Coordinates after removal:", len(results))
                print("bbAreaArr: ", bbAreaArr)
                centroids = np.array(centroids)
                print(centroids)
                D = dist.cdist(centroids, centroids, metric="euclidean")
                print("D:", D)
                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if D[i, j] < MIN_DISTANCE:
                            # update our violation set with the indexes of the centroid pairs
                            serious.add(i)
                            serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                        if (D[i, j] < MAX_DISTANCE) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

                    # loop over the results
                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        # extract the bounding box and centroid coordinates, then
                        # initialize the color of the annotation
                        (startX, startY, endX, endY) = bbox
                        area = ((endX - startX) * (endY - startY)) / (25.4 * 25.4)
                        print("area:", area)
                        print("break.....")
                        if area > 45 and area < 90:
                            (cX, cY) = centroid
                            color = (0, 255, 0)

                            # if the index pair exists within the violation/abnormal sets, then update the color
                            if i in serious:
                                color = (0, 0, 255)
                            elif i in abnormal:
                                color = (0, 255, 255)  # orange = (0, 165, 255)

                            # draw (1) a bounding box around the person and (2) the
                            # centroid coordinates of the person,
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            area = int(((endX - startX) * (endY - startY)) / (
                                        25.4 * 25.4))  ###################################
                            areaArr.append(area)
                            cv2.putText(frame, str(area), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                        (0, 0, 255), 1)
                            cv2.putText(frame, str(bbAreaArr), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                        (0, 0, 255),
                                        1)
                            # cv2.putText()#################################################
                            # print("area of box {0} is {1}".format(i,))
                            # cv2.circle(frame, (cX, cY), 5, color, 2)   # to draw central point

                            # if len(serious) >5:
                            #     huddling = "Huddling Activity Detected: {}".format(len(serious))
                            #     cv2.putText((frame, huddling , (70,60),cv2.FONT_HERSHEY_SIMPLEX, 0.70,(0,0,255),3))
                            # draw some of the parameters
                            Safe_Distance = "Safe distance: >{} px".format(MAX_DISTANCE)
                            # cv2.putText(frame, Safe_Distance, (70, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

                            Threshold = "Threshold limit: {}".format(Threshold)
                            # cv2.putText(frame, Threshold, (70, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

                            # draw the total number of social distancing violations on the output frame    24/48 =  1/2 = 0.5
                            text = "Total serious violations: {}".format(len(serious))
                            # cv2.putText(frame, text, (10, frame.shape[0] - 55),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

                            text1 = "Total abnormal violations: {}".format(len(abnormal))
                            # cv2.putText(frame, text1, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                            # elapsed_time = time.time() - starting_time   # disha
                            # print('execution time1:', elapsed_time)   # disha
                            print("got out...")
                            cv2.putText(frame, text, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                        (0, 0, 255),
                                        2)  # red bounding Box
                            cv2.putText(frame, text1, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                        (0, 250, 255),
                                        2)  # Yellow bounding Box

                            if len(serious) >= 3:
                                text2 = "Huddling Activity Detected"
                                cv2.putText(frame, text2, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                            2)
                                print('huddle', text2)
                            # cv2.putText(frame, text2, (10,frame.shape[0]-115), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

                        print('frameid:', frame_id)
                        elapsed_time = time.time() - starting_time
                        print('execution time2:', elapsed_time)
                        fps = frame_id / elapsed_time
                        print('fps:', fps)
                        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.70,
                                    (0, 0, 250), 2)

            # print("reached here..")
            # print("Memory usage: ", process.memory_info().rss)
            # print("CPU percentage: ",psutil.cpu_percent(interval=0.1))
            # cv2.VideoWriter()

            # cv2.putText(frame, "person detected: " + str(boxes_count), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 1)
            # print("person detected no. :" + str(boxes_count))
            # if boxes_count>5:
            cv2.imshow("Cam{0}".format(), frame)
            cv2.setWindowProperty("Cam1", cv2.WND_PROP_TOPMOST, 10)
            # cv2.imwrite(rootpath + "\\result\\" + fname + "\\" + str(frame_id)+".jpg", frame)
            # if not cv2.imwrite(rootpath + "\\result\\" + fname + "\\" + fname + frame_id + ".jpg", frame):
            out_vid.write(frame)
            frame_id = int(frame_id)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except:
            break
    # videocreate(vidpath, fname)
    cap.release()
    # writer.release()
    cv2.destroyAllWindows()
# firebase = firebase.FirebaseApplication('https://counting-people-dda5f.firebaseio.com/')
# result = firebase.put('/counting-people-dda5f','people',boxes_count)
# print(result)

