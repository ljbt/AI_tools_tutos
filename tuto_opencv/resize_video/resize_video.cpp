#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
VideoCapture videoSource;
Mat frame;
#define VIDEO_PATH "video.mp4"

int main() 
{
    //Open video
    if (!videoSource.open(VIDEO_PATH))
    {            
        std::cout<<"Video not found at "<<VIDEO_PATH<<std::endl;
        return 1;     // Exit if fail
    }
    videoSource.set(CAP_PROP_CONVERT_RGB, 1);
    
    float videoWidth = videoSource.get(CAP_PROP_FRAME_WIDTH);
    float videoHeight = videoSource.get(CAP_PROP_FRAME_HEIGHT);
    float videoAspectRatio = videoWidth / videoHeight;

    std::cout <<"video resolution: " << videoWidth<<", "<<videoHeight<<" aspect ratio: "<<videoAspectRatio<< std::endl;
    
    while(true)
    {
        videoSource >> frame;
        if(frame.empty())
            break;  
        //Resize frame
        resize(frame, frame, cv::Size(320, 320 / videoAspectRatio));
        imshow("frame", frame);
        if(waitKey(24) == 27) // ESC to quit
            break;
    }
   
   return 0;
}