#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "definitions.h"

using namespace cv;
using namespace cv::ml;
using namespace std;


void help()
{
    cout<< "\n--------------------------------------------------------------------------" << endl
        << "This program shows Support Vector Machines for Non-Linearly Separable Data. " << endl
        << "--------------------------------------------------------------------------"   << endl
        << endl;
}

void fill_data_matrix(Mat trainData, Mat *I)
{
    int thick = -1;
    float px, py;
    // Class 1
    for (int i = 0; i < NTRAINING_SAMPLES; i++)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(*I, Point( (int) px,  (int) py ), 3, Scalar(0, 255, 0), thick);
    }
    // Class 2
    for (int i = NTRAINING_SAMPLES; i <2*NTRAINING_SAMPLES; i++)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(*I, Point( (int) px, (int) py ), 3, Scalar(255, 0, 0), thick);
    }
}