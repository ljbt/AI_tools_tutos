#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 

using namespace cv;
using namespace std;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, char** argv) 
{
    // We'll start by loading an image from the drive
    Mat image = imread("./mandala.jpg");

    // We check that our image has been correctly loaded
    if(image.empty()) 
    {
        cout << "Error: the image has been incorrectly loaded." << endl;
        return 0;
    }

    // Then we create a window to display our image
    namedWindow("image");

    // Finally, we display our image and ask the program to wait for a key to be pressed
    imshow("image", image);
    cout << "image type: " << type2str(image.type()) << endl;
    Vec3b pixelColor = image.at<Vec3b>(0,0);
    cout << "Color pixel value:\n" << pixelColor << endl;
    cout << "second value of color pixel:\n" << (int)pixelColor[1] << endl;
    
    /* Maintenant nous pouvons nous intéresser aux pixels de l'image 
     à la ligne r et la colonne c
    Pour cela il y a différents types:
        CV_8UC1 image en niveaux de gris à 1 canal sur 8 bits     
            uchar pixelGrayValue = image.at<uchar>(r,c)
        CV_32FC1 images en niveaux de gris à 1 canal à virgule flottante sur 32 bits
            float pixelGrayValue = image.at<float>(r,c)
        CV_8UC3 images couleur à 3 canaux à 8 bits
            cv::Vec3b pixelColor = image.at<cv::Vec3b>(r,c)
        CV_32FC3 images couleur à 3 canaux en virgule flottante sur 32 bits 
            cv::Vec3f pixelColor = image.at<cv::Vec3f>(r,c)
    -> La fonction type2str ci-dessus return le type de la matrice

    */

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    imshow("gray image", gray_image); 
    cout << "\ngray image type: " << type2str(gray_image.type()) << endl;
    int pixelGray = (int)image.at<uchar>(0,0);
    cout << "Gray pixel value:\n" << pixelGray << endl;



    waitKey();
    return 0;
}