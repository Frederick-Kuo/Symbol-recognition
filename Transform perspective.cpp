#include <stdio.h>
#include <iostream>
#include "opencv_aee.hpp"
#include "main.hpp"     // You can use this file for declaring defined values and functions
#include "pi2c.h"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
void setup(void)
{
    setupCamera(320, 240);  // Enable the camera for OpenCV
    //Pi2c car(0x07); // Configure the I2C interface to the Car as a global variable
}
int SymbolRecognition(Mat frame)
{
    Mat framecopy = frame.clone(); // create a copy of the input frame
    cvtColor(frame, frame, COLOR_BGR2HSV);  //convert image to hsv

    // Define the color range to select pink objects in the image
    Scalar lower_range = Scalar(145,30,30);
    Scalar upper_range = Scalar(165,245,245);

    // Create a binary mask with the selected pink color range
    Mat pink_mask;
    inRange(frame, lower_range, upper_range, pink_mask);
    // Find the contour with the biggest area in the pink mask
    vector<vector<Point> > contours;//outer vector:multiple contours,inner vector: one contour,point structure:coordinate
    vector<Vec4i> hierarchy;//4 values,1:next contour in same hierarchy,2:previous contour at the same hierarchy,3:first child contour,4:parent contour

    findContours(pink_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//retrieves all of the contours and reconstructs a full hierarchy of nested contours;compresses horizontal, vertical, and diagonal segments and leaves only their end points
    if(!contours.empty())
    {
        int savedContour = -1;
        double maxArea = 0.0;
        for (int i = 0; i< contours.size(); i++)//numbers of contours
        {
            double area = contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                savedContour = i;
            }
        // Create a mask from the largest pink contour and draw it onto a copy of the input frame
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);//type of the image as 8-bit single channel grayscale
        drawContours(mask, contours, savedContour, Scalar(255), FILLED, 8);
        Mat masked_image;
        frame.copyTo(masked_image, mask);
        // Find the contours of the pink mask to extract the outermost contour
        vector<vector<Point>> contours2;
        vector<Vec4i> hierarchy2;
        findContours(mask, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE);
        // Find the four corners of the outermost contour using approxPolyDP
        vector<Point> contour_poly;
        approxPolyDP(contours2[0], contour_poly, 3, true);
        vector<Point2i> corners;
        for (int i = 0; i < contour_poly.size(); i++)
        {
            corners.push_back(Point2i(contour_poly[i].x, contour_poly[i].y));
        }
        // Sort the corners in clockwise order, starting from the top left corner
        sort(corners.begin(), corners.end(), [](const Point2i& a, const Point2i& b) {return a.y < b.y;});
        vector<Point2i> corners_top = {corners[0], corners[1]};
        vector<Point2i> corners_bottom = {corners[2], corners[3]};
        sort(corners_top.begin(), corners_top.end(), [](const Point2i& a, const Point2i& b) {return a.x < b.x;});
        sort(corners_bottom.begin(), corners_bottom.end(), [](const Point2i& a, const Point2i& b) {return a.x > b.x;});
        corners = {corners_top[0], corners_top[1], corners_bottom[0], corners_bottom[1]};
        // Define the destination points for the transformation
        int xsize = 350;
        int ysize = 350;
        Point2f dst[4] = {Point2i(0, 0), Point2i(xsize, 0), Point2i(xsize, ysize), Point2i(0, ysize)};
        // Perform the perspective transformation
        framecopy = transformPerspective(corners, framecopy, xsize, ysize);
        // Compare the symbol image with a set of predefined symbols to recognize the symbol
        // ...
        //cvtColor (framecopy, framecopy, COLOR_BGR2GRAY);
        if (framecopy.cols > 0 && framecopy.rows > 0)
        {
            imshow("Output", framecopy);
            Mat Circle = imread("/home/pi/Desktop/symbol pics/Circle (Red Line).png");
            Mat Star = imread("/home/pi/Desktop/symbol pics/Star (Green Line).png");
            Mat Triangle = imread("/home/pi/Desktop/symbol pics/Triangle (Blue Line).png");
            Mat Umbrella = imread("/home/pi/Desktop/symbol pics/Umbrella (Yellow Line).png");
            cvtColor (Circle, Circle, COLOR_BGR2GRAY);
            cvtColor (Star, Star, COLOR_BGR2GRAY);
            cvtColor (Triangle,Triangle, COLOR_BGR2GRAY);
            cvtColor (Umbrella, Umbrella, COLOR_BGR2GRAY);
            cvtColor (framecopy, framecopy, COLOR_BGR2GRAY);
			threshold(framecopy, framecopy, 128, 255, THRESH_BINARY); // apply binary threshold to grayscale image
			threshold(Circle, Circle, 128, 255, THRESH_BINARY);
			threshold(Star, Star, 128, 255, THRESH_BINARY);
			threshold(Triangle, Triangle, 128, 255, THRESH_BINARY);
			threshold(Umbrella, Umbrella, 128, 255, THRESH_BINARY);
            //equalizeHist(framecopy, framecopy);
            std::cout << "contour detected successfully" << std::endl;
            if(compareImages(framecopy,Circle)>60)
                std::cout << "circle" << std::endl;
            else if(compareImages(framecopy,Star)>60)
                std::cout << "star" << std::endl;
            else if(compareImages(framecopy,Triangle)>60)
                std::cout << "triangle" << std::endl;
            else if(compareImages(framecopy,Umbrella)>60)
                std::cout << "umbrella" << std::endl;
            else
                std::cout << "no matchings found" << std::endl;
            int umb=compareImages(framecopy,Umbrella);
            std::cout << "umbrella matching: " << umb << std::endl;
        }
        else
            std::cout << "contour size zero or negative" << std::endl;
    }
    else
        std::cout << "contour is empty" << std::endl;
    return 0;
    }
int main (int argc, char** argv)
{
    setup();    // Call a setup function to prepare IO and devices
	namedWindow("Photo");   // Create a GUI window called photo
    int result;
	Mat frame,frameHSV;
	while(1)    // Main loop to perform image processing
	{
		frame = captureFrame(); // Capture a frame from the camera and store in a new matrix variable
		flip(frame,frame,-1);
		cvtColor(frame, frameHSV, COLOR_BGR2HSV); // Convert to HSV
		vector<Mat> channels; // Array for channels
		split(frameHSV, channels); // Split the HSV into separate channels
		equalizeHist(channels[2], channels[2]); // Equalise the Value channel
		/*// Apply binary thresholding
		threshold(channels[2], channels[2], 100, 255, THRESH_BINARY);
		// Define kernel for morphological operations
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		// Apply opening operation to remove noise
		morphologyEx(channels[2], channels[2], MORPH_OPEN, kernel);
		// Apply closing operation to enhance edges
		morphologyEx(channels[2], channels[2], MORPH_CLOSE, kernel);
		merge(channels, frameHSV); // Merge back into a single image
		GaussianBlur(frameHSV, frameHSV, Size(11,11), 0, 0);*/
		merge(channels, frameHSV); // Merge back into a single image
		cvtColor(frameHSV, frame,COLOR_HSV2BGR); // Convert back to BGR
		imshow("camera",frame);
		result = SymbolRecognition(frame);
		int key = waitKey(1);   // Wait 1ms for a keypress (required to update windows)
		key = (key==255) ? -1 : key;    // Check if the ESC key has been pressed
		if (key == 27)
			break;
	}
	waitKey(0);
	closeCV();  // Disable the camera and close any windows
	return 0;
}
