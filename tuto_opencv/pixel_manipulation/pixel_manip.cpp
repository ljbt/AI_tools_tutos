#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;

int main(int argc, char **argv)
{

    // Create a container
    Mat im; 

    //Create a vector
    Vec3b *vec;

    // Create an mat iterator
    MatIterator_<Vec3b> it;

    // Read the image in color format
    im = imread("mandala.jpg", 1);

    // iterate through each pixel
    for(it = im.begin<Vec3b>(); it != im.end<Vec3b>(); ++it)
    {
        // Erase the green and red channels 
        (*it)[1] = 0;
        (*it)[2] = 0;
    }

    // Show the image
    imshow("Image with only blue component", im);


    for(int r = 0; r < im.rows; r++)
    {
        vec = im.ptr<Vec3b>(r); //pointer on row

        for(int c = 0; c < im.cols; c++)
        {
            vec[c] = Vec3b(vec[c][2],vec[c][0],vec[c][1]); // B G R
        }
    }

    imshow("Image transformed with pointer", im);

    // Wait for a key
    waitKey(0);

    return 0;
}