#include "DetectionModelTRT.h"
#include "Timers.h"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <vector>

#define BATCH_SIZE 16

using namespace std;


void detectOnRandomImages(
    const string& inputImageFolder, 
    const string& outputImageFolder,
    DetectionModelTRT& engine,
    Params p,
    float confThreshold,
    int batchSize
){
    
    const filesystem::path img_folder{inputImageFolder};

    vector<cv::Mat> preprocessedImgList;
    for (auto const& dir_entry : filesystem::directory_iterator{img_folder}){
        if (filesystem::is_directory(dir_entry)) continue;
        
        cv::Mat image = cv::imread(dir_entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) continue;
        preprocessedImgList.push_back(Utility::processMat(p, image));

    }

    int numberOfImages = preprocessedImgList.size();
    cout << "Total number of Images in directory: " << numberOfImages << endl;


    mt19937 randomRange(random_device{}());
    Timer timer;
    float* rawOutput = nullptr;

    vector<cv::Mat> randomBatch(batchSize);
    sample(preprocessedImgList.begin(), preprocessedImgList.end(), randomBatch.begin(), batchSize, randomRange);

    timer.tic();
    engine.detect(randomBatch, rawOutput);
    double diff = timer.toc();
    cout << "Batch size=" << batchSize << " took " << diff  << " ms, "  <<
        diff/batchSize << " ms/img" << endl;

    vector<vector<Detection>> resultList = Utility::processOutput(rawOutput, batchSize, p, confThreshold);
        
    for(int i = 0; i < batchSize; ++i){
        auto img = randomBatch[i];
        auto result = resultList[i];

        string filename = outputImageFolder + to_string(i) + ".png";
        cv::Mat resultImage = Utility::drawResult(img, result);
        cv::imwrite(filename, resultImage);
    }
}

void detectOnVideo(
    const string& inputVideoPath, 
    const string& outputVideoPath, 
    DetectionModelTRT& engine, 
    Params p,
    float confThreshold
) {
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) throw runtime_error("Could not open video file.");

    cv::Mat frame;
    vector<cv::Mat> framesBatch;

    cv::VideoWriter writer(outputVideoPath, 
                       static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)), 
                       cap.get(cv::CAP_PROP_FPS),
                       cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    
    int currentBS;
    vector<cv::Mat> preprocessedFrames;
    vector<cv::Mat> processedFrames;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        framesBatch.emplace_back(frame.clone());
        if (framesBatch.size() >= BATCH_SIZE || !cap.read(frame)) {    
            currentBS = framesBatch.size();

            preprocessedFrames.reserve(currentBS);
            processedFrames.reserve(currentBS);

            for (const auto& frame: framesBatch){
                preprocessedFrames.push_back(Utility::processMat(p, frame));
            }
            float* rawOutput = nullptr;
            engine.detect(preprocessedFrames, rawOutput);

            vector<vector<Detection>> resultList = Utility::processOutput(
                rawOutput, preprocessedFrames.size(), p, confThreshold
            );

            for (size_t i = 0; i < currentBS && i < resultList.size(); ++i) {
                processedFrames.push_back(
                    Utility::drawResult(framesBatch[i], resultList[i])
                );
            }

            for (const auto& img : processedFrames) {
                writer.write(img);
            }

            framesBatch.clear();
            preprocessedFrames.clear();
            processedFrames.clear();
            delete[] rawOutput;
        }
    }

    writer.release();
}


int main(int argc, char** argv)
{
    char* onnxFileName = argv[1];

    filesystem::path onnxFilePath(onnxFileName);
  
    Params params = Utility::createDefaultParams(onnxFileName);
    cout << "Building Engine with params:\n"
          << "- ONNX file path: " << params.onnxFileName << "\n"
          << "- Engine file name: " << params.engineFileName << "\n"
          << "- Optimizations: INT8=" << boolalpha << params.int8 << " "
          << "FP16=" << boolalpha << params.fp16 << " "
          << "BF16=" << boolalpha << params.bf16 << "\n"
          << "- Input (HxWxC): " << params.inputHeight << "x" << params.inputWidth << "x" << params.inputNChannels << "\n"
          << "- Output: " << params.outputLength << "x" << params.outputItemSize << "\n"
          << "- Calibration data path: " << params.calibrationDataPath << "\n"
          << "- Calibration cache file: " << params.calibrationCacheFile << "\n";
    DetectionModelTRT Engine(params);

    bool status = Engine.prepareEngine();
    if (!status) {
        cerr << "Failed to prepare engine" << endl;
        return 1;
    }
    
    // detectOnRandomImages(
    //     "assets/",
    //     "results/trt/",
    //     Engine,
    //     params,
    //     0.6,
    //     10
    // );

    detectOnVideo(
        "videos/kittens.mov",
        "videos/processed.mov",
        Engine,
        params, 
        0.5
    );
    
    return 0;
}