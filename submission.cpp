/*
 * File:      submission.cpp
 * Author:    Richard Purcell
 * Date:      2020-22-12
 * Version:   1.0
 * Purpose:   Given an image add virtual makeup and display the altered image.
 * Usage:     $ ./submission
 * Notes:     Created for OpenCV's Computer Vision 2 Project 1.
 *            Provided filenames are hard coded for this project.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <./dlib/opencv.h>
#include <./dlib/image_processing.h>
#include <./dlib/image_processing/frontal_face_detector.h>
#include <./dlib/image_io.h>
#include <./dlib/gui_widgets.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO 1

//int selectedpoints[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 58, 59};
int selectedpoints[] = {36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

std::vector<int> selectedIndex(selectedpoints, selectedpoints + sizeof(selectedpoints) / sizeof(int));

//read points stored in text files
std::vector<Point2f> getSavedPoints(string pointsFileName)
{
    std::vector<Point2f> points;
    ifstream ifs(pointsFileName.c_str());
    float x, y;
    if (!ifs)
        cout << "Unable to open file " << pointsFileName << endl;
    else
    {
        cout << "Successfully opened " << pointsFileName << endl;
    }
    
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));
    }
    return points;
}

int
main()
{
    string modelPath = "./models/shape_predictor_68_face_landmarks.dat";
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize(modelPath) >> predictor;

    string overlayFile = "./images/eye_makeup1.png";
    string imageFile = "./images/ted_cruz.jpg";

    //Read eye image
    Mat eyes, targetImage, eyesAlphaMask;
    Mat imgWithMask = imread(overlayFile, IMREAD_UNCHANGED);
    std::vector<Mat> rgbaChannels(4);

    //Split into channels
    split(imgWithMask, rgbaChannels);

    //Extract eyes image
    std::vector<Mat> bgrchannels;
    bgrchannels.push_back(rgbaChannels[0]);
    bgrchannels.push_back(rgbaChannels[1]);
    bgrchannels.push_back(rgbaChannels[2]);

    merge(bgrchannels, eyes);
    eyes.convertTo(eyes, CV_32F, 1.0/255.0);

    //Extract the beard mask
    std::vector<Mat> maskchannels;
    maskchannels.push_back(rgbaChannels[3]);
    maskchannels.push_back(rgbaChannels[3]);
    maskchannels.push_back(rgbaChannels[3]);

    merge(maskchannels, eyesAlphaMask);
    eyesAlphaMask.convertTo(eyesAlphaMask, CV_32FC3);

    //Read points from eye file
    std::vector<Point2f> featurePoints1 = getSavedPoints("./images/eye_makeup1.png.txt");

    //Calculate Delaunay triangles
    Rect rect = boundingRect(featurePoints1);
    std::vector<std::vector<int> >dt;
    calculateDelaunayTriangles(rect, featurePoints1, dt);

    //Get the face image
    targetImage = imread(imageFile);
    //int height = targetImage.rows;
    //float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
    //cv::resize(targetImage, targetImage, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

    std::vector<Point2f> points2 = getSavedPoints("./images/ted_cruz.jpg.txt");
    //std::vector<Point2f> points2 = getLandmarks(detector, predictor, targetImage, (float) FACE_DOWNSAMPLE_RATIO);
    cout << "at line 108" << endl;
    std::vector<Point2f> featurePoints2;
    for(int i=0; i<selectedIndex.size(); i++)
    {
        featurePoints2.push_back(points2[selectedIndex[i]]);
        constrainPoint(featurePoints2[i], targetImage.size());
    }
    cout << "at line 115" << endl;
    //convert Mat to float data type
    targetImage.convertTo(targetImage, CV_32F, 1.0/255.0);

    //empty warp image
    Mat eyesWarped = Mat::zeros(targetImage.size(), eyes.type());
    Mat eyesAlphaMaskWarped = Mat::zeros(targetImage.size(), eyesAlphaMask.type());

    cout << "at line 123" << endl;
    //Apply affine transformation to Delaunay triangles
    for(size_t i=0; i<dt.size(); i++)
    {
        std::vector<Point2f> t1, t2;
        //Get points for img1, targetImage corresponding to the triangles
        for(size_t j=0; j<3; j++)
        {
            cout << "i is " << i << " j is " << j << endl;
            t1.push_back(featurePoints1[dt[i][j]]);
            t2.push_back(featurePoints2[dt[i][j]]);
        }
        warpTriangle(eyes, eyesWarped, t1, t2);
        warpTriangle(eyesAlphaMask, eyesAlphaMaskWarped, t1, t2);
    }


    
    Mat mask1;
    eyesAlphaMaskWarped.convertTo(mask1, CV_32FC3, 1.0/255.0);

    Mat mask2 = Scalar(1.0,1.0,1.0) - mask1;
    Mat temp1 = targetImage.mul(mask2);
    Mat temp2 = eyesWarped.mul(mask1);

    Mat result = temp1 + temp2;

    imshow("Display window", result);
    imshow("eye mask", eyesAlphaMask);
    int k = waitKey(0);

    return 0;
}