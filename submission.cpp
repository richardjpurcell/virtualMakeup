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

int selectedpoints[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
                        29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                        54,55,56,57,58,59,60,61,62,63,64,65,66,67};

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

    string target_img_file = "./images/ted_cruz.jpg";
    string face_01_img_file = "./images/face_01_img.png";
    string eyes_01_msk_file = "./images/eyes_01_msk.png";
    string lips_01_msk_file = "./images/lips_01_msk.png";
    string chks_01_msk_file = "./images/chks_01_msk.png";


    //Read  images
    Mat target_img = imread(target_img_file, IMREAD_COLOR);
    Mat face_01_img = imread(face_01_img_file, IMREAD_COLOR);
    Mat eyes_01_msk = imread(eyes_01_msk_file,IMREAD_COLOR);
    Mat lips_01_msk = imread(lips_01_msk_file, IMREAD_COLOR);
    Mat chks_01_msk = imread(chks_01_msk_file, IMREAD_COLOR);

    face_01_img.convertTo(face_01_img, CV_32F, 1.0/255.0);
    eyes_01_msk.convertTo(eyes_01_msk, CV_32F, 1.0/255.0);
    lips_01_msk.convertTo(lips_01_msk, CV_32F, 1.0/255.0);
    chks_01_msk.convertTo(chks_01_msk, CV_32F, 1.0/255.0);

    //Read points from eye file
    std::vector<Point2f> featurePoints1 = getSavedPoints("./images/face_01_img_0.txt");

    //Calculate Delaunay triangles
    Rect rect = boundingRect(featurePoints1);
    std::vector<std::vector<int> >dt;
    calculateDelaunayTriangles(rect, featurePoints1, dt);

    std::vector<Point2f> points2 = getSavedPoints("./images/ted_cruz.jpg.txt");

    std::vector<Point2f> featurePoints2;
    for(int i=0; i<selectedIndex.size(); i++)
    {
        featurePoints2.push_back(points2[selectedIndex[i]]);
        constrainPoint(featurePoints2[i], target_img.size());
    }

    //result images
    Mat face_01_img_warped = Mat::zeros(target_img.size(), face_01_img.type());
    Mat eyes_01_msk_warped = Mat::zeros(target_img.size(), eyes_01_msk.type());
    Mat lips_01_msk_warped = Mat::zeros(target_img.size(), lips_01_msk.type());
    Mat chks_01_msk_warped = Mat::zeros(target_img.size(), chks_01_msk.type());
    Mat result;

    //Apply affine transformation to Delaunay triangles
    for(size_t i=0; i<dt.size(); i++)
    {
        std::vector<Point2f> t1, t2;

        for(size_t j=0; j<3; j++)
        {
            t1.push_back(featurePoints1[dt[i][j]]);
            t2.push_back(featurePoints2[dt[i][j]]);
        }

        warpTriangle(face_01_img, face_01_img_warped, t1, t2);
        warpTriangle(eyes_01_msk, eyes_01_msk_warped, t1, t2);
        warpTriangle(lips_01_msk, lips_01_msk_warped, t1, t2);
        warpTriangle(chks_01_msk, chks_01_msk_warped, t1, t2);
    }
  
    //prepare warped images and masks for seamless cloning onto target image
    face_01_img_warped = face_01_img_warped * 255;
    face_01_img_warped.convertTo(face_01_img_warped, CV_8U);
    cv::rectangle(face_01_img_warped, Point(0,0), Point(600,800), Scalar(255,0,0), 5);

    eyes_01_msk_warped = eyes_01_msk_warped * 255;
    eyes_01_msk_warped.convertTo(eyes_01_msk_warped, CV_8U);
    cv::rectangle(eyes_01_msk_warped, Point(0,0), Point(600,800), Scalar(255,0,0), 5);

    lips_01_msk_warped = lips_01_msk_warped * 255;
    lips_01_msk_warped.convertTo(lips_01_msk_warped, CV_8U);
    cv::rectangle(lips_01_msk_warped, Point(0,0), Point(600,800), Scalar(255,0,0), 5);

    chks_01_msk_warped = chks_01_msk_warped * 255;
    chks_01_msk_warped.convertTo(chks_01_msk_warped, CV_8U);
    cv::rectangle(chks_01_msk_warped, Point(0,0), Point(600,800), Scalar(255,0,0), 5);

    //combine masks
    eyes_01_msk_warped = eyes_01_msk_warped + lips_01_msk_warped + chks_01_msk_warped;

    //seamless clone warped face onto target face
    Point center(target_img.cols/2, target_img.rows/2);
    seamlessClone(face_01_img_warped, target_img, eyes_01_msk_warped, center, result, MIXED_CLONE);

    //display images
    imshow("Original Image", target_img);   
    imshow("Result Image", result);
    int k = waitKey(0);

    return 0;
}