#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "inference.h"
#include "stalker.h"

using namespace std;
using namespace cv;


uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


int  readOnImage(){

    std::string projectBasePath = "/home/halilerden/Documents/workFiles/06-imageProcess/imageProcess/opencv-yolo"; // Set your ultralytics base path

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolo11s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/onurAmirim1.png");
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    for (int i = 0; i < imageNames.size(); ++i)
    {
        cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }
    return 0;

}

int main(int argc, char **argv)
{

        Stalker stalker;

    std::string projectBasePath = "/home/halilerden/Documents/workFiles/06-imageProcess/imageProcess/opencv-yolo"; // Set your ultralytics base path

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //



    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolov9s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    Mat videoFrame;
    // VideoCapture camera(0, CAP_V4L2);
    VideoCapture camera("/home/halilerden/Documents/workFiles/06-imageProcess/testVideo/Front_View.mp4");

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }
    camera.set(CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CAP_PROP_FRAME_HEIGHT, 480);
    // if (camera.set(CAP_PROP_FPS, 30.0))
    {
        std::cout << "camera setted 30 fps" << std::endl;
    }
    auto lapse = timeSinceEpochMillisec();

    while (camera.read(videoFrame)) {
        auto start = timeSinceEpochMillisec();
        Mat frame;
        Size inputSize(640,640);
        resize(videoFrame,frame,inputSize);
        cvtColor(videoFrame,videoFrame,COLOR_BGR2GRAY);
        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;
        // Detected bounding box'ları topluyoruz.
        std::vector<cv::Rect> detectedBoxes;
        for (int i = 0; i < detections; ++i) {
            detectedBoxes.push_back(output[i].box);
        }

        // Stalker ile bounding box'ları işliyoruz.
        std::vector<cv::Rect> trackedBoxes = stalker.processDetections(detectedBoxes);

        int carCountInPicture=0;
        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            if(detection.class_id==2){
                carCountInPicture++;
            }

            // // Detection box text
            // std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4)+ ' '+ std::to_string(carCountInPicture).substr(0, 4);
            // cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            // cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            // // std::cout << "class type:" << classString << std::endl;

            // Takip bilgisi ve tespit bilgisi yazdır.
            std::string classString = output[i].className + " " +
                                      std::to_string(output[i].confidence).substr(0, 4) +
                                      " ID: " + std::to_string(i);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);


            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);


        }
        std::string carString = "arac Sayisi :" + std::to_string(carCountInPicture);

        cv::putText(frame,carString,cv::Point(100 , 50 ), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 100, 0), 2, 0);


        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);
        std::cout << timeSinceEpochMillisec()-start << "milisecond process" << std::endl;
        std::cout << timeSinceEpochMillisec()-lapse << "milisecond lapsed" << std::endl;
        lapse = timeSinceEpochMillisec();
        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }

    }

    camera.release();
    destroyAllWindows();
    return 0;
}
