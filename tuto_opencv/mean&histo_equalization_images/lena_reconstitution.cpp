#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main() 
{
    Mat img1,img2;
    img1 = imread("lena1.jpg",IMREAD_GRAYSCALE);
    img2 = imread("lena2.jpg",IMREAD_GRAYSCALE);

    if( ! img1.data || ! img2.data)
    {
        cout << "Error loading images"<< endl;
        return -1;
    }
/*     imshow("lena 1", img1);
    imshow("lena 2", img2); */
    // convert images from gray 8 bit int, to gray 32 bit float for better computations
    img1.convertTo(img1,CV_32FC1); 
    img2.convertTo(img2,CV_32FC1);

    Mat img3 = (img1+img2)/2.; //mean value between img1 and img2
    
    // convert images from gray 8 bit int, to gray 32 bit float for better computations
    img1.convertTo(img1,CV_8UC1); 
    img2.convertTo(img2,CV_8UC1);
    img3.convertTo(img3,CV_8UC1);

    imshow("lena 1", img1);
    imshow("lena 2", img2);
    imshow("mean image", img3);

    Mat hist;
    equalizeHist(img3,hist);
    imshow("after hist equalization", hist);

    waitKey(0);
    destroyAllWindows();
    return 0;
}