#ifndef STALKER_H
#define STALKER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Struct to hold tracker information
struct TrackedObject {
    int id;
    cv::Rect boundingBox;
    cv::KalmanFilter kalmanFilter;
    int missedFrames;
};

class Stalker {
public:
    Stalker();

    // Function to process detected inputs and update tracking
    std::vector<cv::Rect> processDetections(const std::vector<cv::Rect>& detections);

private:
    std::vector<TrackedObject> trackers;
    int nextId;

    // Helper functions
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    cv::KalmanFilter createKalmanFilter();
    void addNewTrackers(const std::vector<cv::Rect>& detections, const std::vector<bool>& matchedDetections);
    void updateTrackers(const std::vector<cv::Rect>& detections);
    void correctPosition(cv::KalmanFilter& kalmanFilter, const cv::Rect& detection) ;
    cv::Rect predictNextPosition(cv::KalmanFilter& kalmanFilter);
    void removeLostTrackers();
};

#endif // STALKER_H
