#include "stalker.h"

Stalker::Stalker() {}

std::vector<cv::Rect> Stalker::processDetections(const std::vector<cv::Rect> &detections)
{
    updateTrackers(detections);
    std::vector<bool> matchedDetections(detections.size(), false);

    // Yeni takip nesneleri ekle
    addNewTrackers(detections, matchedDetections);

    // Eşleşmeyen nesneleri kaldır
    removeLostTrackers();

    // Güncellenen bounding box'ları döndür
    std::vector<cv::Rect> updatedBoxes;
    for (const auto& tracker : trackers) {
        updatedBoxes.push_back(tracker.boundingBox);
    }
    return updatedBoxes;
}

float Stalker::calculateIoU(const cv::Rect &box1, const cv::Rect &box2)
{
    int x1 = std::max(box1.x, box2.x); // Kesişim dikdörtgeninin sol üst köşe X'i
    int y1 = std::max(box1.y, box2.y); // Kesişim dikdörtgeninin sol üst köşe Y'si
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width); // Sağ alt X
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height); // Sağ alt Y

    // Kesişim alanını hesapla
    int intersectionWidth = std::max(0, x2 - x1);
    int intersectionHeight = std::max(0, y2 - y1);
    int intersectionArea = intersectionWidth * intersectionHeight;

    // Birleşim alanını hesapla
    int unionArea = box1.area() + box2.area() - intersectionArea;

    // IoU'yu döndür
    return static_cast<float>(intersectionArea) / unionArea;


}

cv::KalmanFilter Stalker::createKalmanFilter()
{
    // 4 durum değişkeni (x, y, genişlik, yükseklik), 2 ölçüm (x, y)
    cv::KalmanFilter kf(4, 2, 0);

    // Durum geçiş matrisi (x, y, genişlik, yükseklik tahmini)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);

    // Ölçüm matrisi
    kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);

    // Süreç gürültü kovaryans matrisi (Tahmin doğruluğu artırmak için ayarlanabilir)
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.1;

    // Ölçüm gürültü kovaryans matrisi
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 0.1;

    // Başlangıç hata kovaryans matrisi
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    // Başlangıç durumu
    kf.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);

    return kf;


}

void Stalker::addNewTrackers(const std::vector<cv::Rect> &detections, const std::vector<bool> &matchedDetections)
{
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matchedDetections[i]) {
            bool foundSimilar = false;
            for (const auto& tracker : trackers) {
                float iou = calculateIoU(tracker.boundingBox, detections[i]);
                if (iou > 0.1) {
                    foundSimilar = true;
                    break;
                }
            }
            if (!foundSimilar) { // Yeterince benzer nesne bulunmadığında ekle
                cv::KalmanFilter kf = createKalmanFilter();
                kf.statePost.at<float>(0) = detections[i].x;
                kf.statePost.at<float>(1) = detections[i].y;
                kf.statePost.at<float>(2) = detections[i].width;
                kf.statePost.at<float>(3) = detections[i].height;

                trackers.push_back({nextId++, detections[i], kf, 0});
            }
        }
    }
}

void Stalker::updateTrackers(const std::vector<cv::Rect> &detections)
{
    std::vector<bool> matchedDetections(detections.size(), false); // Tespit eşleşme durumu

    for (auto& tracker : trackers) {
        // Kalman filtresi ile tahmin edilen konumu al
        cv::Rect predictedBox = predictNextPosition(tracker.kalmanFilter);

        float maxIoU = 0.0;
        int bestMatchIndex = -1;

        // Her tespit için IoU hesapla
        for (size_t i = 0; i < detections.size(); ++i) {
            if (matchedDetections[i]) continue; // Zaten eşleştiyse atla

            float iou = calculateIoU(predictedBox, detections[i]);
            if (iou > maxIoU) {
                maxIoU = iou;
                bestMatchIndex = i;
            }
        }

        // IoU eşiğini kontrol et
        if (maxIoU > 0.3 && bestMatchIndex != -1) {
            correctPosition(tracker.kalmanFilter, detections[bestMatchIndex]); // Kalman düzeltme
            tracker.boundingBox = detections[bestMatchIndex]; // Güncel konum
            tracker.missedFrames = 0; // Eşleşme bulundu, kaybolma sıfırla
            matchedDetections[bestMatchIndex] = true; // Bu tespit eşleşti
        } else {
            // Eşleşme yok, sadece tahmini güncelle
            tracker.boundingBox = predictedBox;
            tracker.missedFrames++;
        }
    }
}

void Stalker::correctPosition(cv::KalmanFilter &kalmanFilter, const cv::Rect &detection)
{
    cv::Mat measurement(2, 1, CV_32F); // Ölçüm matrisini oluştur
    measurement.at<float>(0) = detection.x; // Ölçüm: x konumu
    measurement.at<float>(1) = detection.y; // Ölçüm: y konumu

    kalmanFilter.correct(measurement); // Tahmini güncelle
}

cv::Rect Stalker::predictNextPosition(cv::KalmanFilter &kalmanFilter)
{
    cv::Mat prediction = kalmanFilter.predict(); // Tahmini hesapla
    int x = static_cast<int>(prediction.at<float>(0)); // x konumu
    int y = static_cast<int>(prediction.at<float>(1)); // y konumu
    int width = static_cast<int>(prediction.at<float>(2)); // genişlik
    int height = static_cast<int>(prediction.at<float>(3)); // yükseklik

    return cv::Rect(x, y, width, height); // Tahmin edilen kutu
}

void Stalker::removeLostTrackers()
{
    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                  [](const TrackedObject& tracker) {
                                      return tracker.missedFrames > 10; // Eşik değer (ör: 10 kare)
                                  }),
                   trackers.end());
}
