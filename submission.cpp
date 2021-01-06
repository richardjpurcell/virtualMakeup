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

#define WRITE_BASE_LANDMARKS 0
#define WRITE_OVERLAY_LANDMARKS 1
#define MAKEUP_STYLE FASHION_FACE
string base_img_file = "./images/girl-no-makeup.jpg";

enum {DEFAULT_FACE, FASHION_FACE, CLOWN_FACE, GOTH_FACE};

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

void 
writeLandmarksToFile(full_object_detection &landmarks, const string &filename)
{
    //open file
    std::ofstream ofs;
    ofs.open(filename);

    //loop over all landmark poinCLOWNts
    for(int i=0; i<landmarks.num_parts(); i++)
    {
        //print x and y coordinates to file
        ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << endl;
    }

    //close file
    ofs.close();
}

void 
landmarkDetect(const cv::Mat& img, string landmarksBasename)
{
    string modelPath = "./models/shape_predictor_68_face_landmarks.dat";
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(modelPath) >> landmarkDetector;
    //convert openCV image to Dlib image
    cv_image<bgr_pixel> dlibIm(img);
    //detect face in img
    std::vector<dlib::rectangle> faceRect = faceDetector(dlibIm);
    //detect landmarks on face if found
    full_object_detection landmarks = landmarkDetector(dlibIm, faceRect[0]);
    //write landmarks to disk
    std::stringstream landmarksFilename;
    landmarksFilename << landmarksBasename << ".txt";
    cout << "Saving  landmarks to " << landmarksFilename.str() << endl;
    writeLandmarksToFile(landmarks, landmarksFilename.str());
}


int
main()
{
    std::stringstream overlay_img_file;
    std::stringstream eyes_msk_file;
    std::stringstream lips_msk_file;
    std::stringstream chks_msk_file;
    overlay_img_file << "./images/face_" << MAKEUP_STYLE << "_img.png";
    eyes_msk_file << "./images/eyes_" << MAKEUP_STYLE << "_msk.png";
    lips_msk_file << "./images/lips_" << MAKEUP_STYLE << "_msk.png";
    chks_msk_file << "./images/chks_" << MAKEUP_STYLE << "_msk.png";

    //Read  images
    Mat base_img = imread(base_img_file, IMREAD_COLOR);
    Mat overlay_img = imread(overlay_img_file.str(), IMREAD_COLOR);
    Mat eyes_msk = imread(eyes_msk_file.str(),IMREAD_COLOR);
    Mat lips_msk = imread(lips_msk_file.str(), IMREAD_COLOR);
    Mat chks_msk = imread(chks_msk_file.str(), IMREAD_COLOR);

    //write landmark points to file
    if(WRITE_BASE_LANDMARKS) { landmarkDetect(base_img, base_img_file); }
    if(WRITE_OVERLAY_LANDMARKS) { landmarkDetect(overlay_img, overlay_img_file.str()); }

    overlay_img.convertTo(overlay_img, CV_32F, 1.0/255.0);
    eyes_msk.convertTo(eyes_msk, CV_32F, 1.0/255.0);
    lips_msk.convertTo(lips_msk, CV_32F, 1.0/255.0);
    chks_msk.convertTo(chks_msk, CV_32F, 1.0/255.0);

        //result images
    Mat overlay_img_warped = Mat::zeros(base_img.size(), overlay_img.type());
    Mat eyes_msk_warped = Mat::zeros(base_img.size(), eyes_msk.type());
    Mat lips_msk_warped = Mat::zeros(base_img.size(), lips_msk.type());
    Mat chks_msk_warped = Mat::zeros(base_img.size(), chks_msk.type());
    Mat result;



    //Read landmark points
    std::stringstream featurePoints1_file;
    std::stringstream featurePoints2_file;
    featurePoints1_file << overlay_img_file.str() << ".txt";
    featurePoints2_file << base_img_file << ".txt";
    std::vector<Point2f> featurePoints1 = getSavedPoints(featurePoints1_file.str());
    std::vector<Point2f> points2 = getSavedPoints(featurePoints2_file.str());
    std::vector<Point2f> featurePoints2;
    for(int i=0; i<selectedIndex.size(); i++)
    {
        featurePoints2.push_back(points2[selectedIndex[i]]);
        constrainPoint(featurePoints2[i], base_img.size());
    }

    //Calculate Delaunay triangles
    Rect rect = boundingRect(featurePoints1);
    std::vector<std::vector<int> >dt;
    calculateDelaunayTriangles(rect, featurePoints1, dt);

    //Apply affine transformation to Delaunay triangles
    for(size_t i=0; i<dt.size(); i++)
    {
        std::vector<Point2f> t1, t2;

        for(size_t j=0; j<3; j++)
        {
            t1.push_back(featurePoints1[dt[i][j]]);
            t2.push_back(featurePoints2[dt[i][j]]);
        }

        warpTriangle(overlay_img, overlay_img_warped, t1, t2);
        warpTriangle(eyes_msk, eyes_msk_warped, t1, t2);
        warpTriangle(lips_msk, lips_msk_warped, t1, t2);
        warpTriangle(chks_msk, chks_msk_warped, t1, t2);
    }

    int WIDTH = base_img.cols;
    int HEIGHT = base_img.rows;
  
    //prepare warped images and masks for seamless cloning onto target image
    overlay_img_warped = overlay_img_warped * 255;
    overlay_img_warped.convertTo(overlay_img_warped, CV_8U);
    cv::rectangle(overlay_img_warped, Point(0,0), Point(WIDTH,HEIGHT), Scalar(255,0,0), 5);

    eyes_msk_warped = eyes_msk_warped * 255;
    eyes_msk_warped.convertTo(eyes_msk_warped, CV_8U);
    cv::rectangle(eyes_msk_warped, Point(0,0), Point(WIDTH,HEIGHT), Scalar(255,0,0), 5);

    lips_msk_warped = lips_msk_warped * 255;
    lips_msk_warped.convertTo(lips_msk_warped, CV_8U);
    cv::rectangle(lips_msk_warped, Point(0,0), Point(WIDTH,HEIGHT), Scalar(255,0,0), 5);

    chks_msk_warped = chks_msk_warped * 255;
    chks_msk_warped.convertTo(chks_msk_warped, CV_8U);
    cv::rectangle(chks_msk_warped, Point(0,0), Point(WIDTH,HEIGHT), Scalar(255,0,0), 5);

    //combine masks
    eyes_msk_warped = eyes_msk_warped + lips_msk_warped + chks_msk_warped;

    //seamless clone warped face onto target face
    Point center(base_img.cols/2, base_img.rows/2);
    seamlessClone(overlay_img_warped, base_img, eyes_msk_warped, center, result, NORMAL_CLONE);

    //display images
    imshow("mask", eyes_msk_warped);
    imshow("Original Image", base_img);   
    imshow("Result Image", result);
    int k = waitKey(0);

    return 0;
}