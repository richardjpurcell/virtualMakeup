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
#include "./renderFace.hpp"


using namespace std;
using namespace cv;
using namespace dlib;



void 
writeLandmarksToFile(full_object_detection &landmarks, const string &filename)
{
    //open file
    std::ofstream ofs;
    ofs.open(filename);

    //loop over all landmark points
    for(int i=0; i<landmarks.num_parts(); i++)
    {
        //print x and y coordinates to file
        ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << endl;
    }

    //close file
    ofs.close();
}

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
    //get face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();
    //landmark detector is implemented in the shape_predictor class
    shape_predictor landmarkDetector;
    string PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat";
    deserialize(PREDICTOR_PATH) >> landmarkDetector;

    //read image
    string imageFilename("./images/reference_face.png");
    cv::Mat img = cv::imread(imageFilename);

    string landmarksBasename("./images/reference_face");

    //convert openCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(img);

    //detect faces in the image
    std::vector<dlib::rectangle> faceRects = faceDetector(dlibIm);
    cout << "Number of faces detected: " << faceRects.size() << endl;

    //vector to store landmarks of all detected faces
    std::vector<full_object_detection> landmarksAll;

    //loop over all detected face rectangles
    for (int i=0; i<faceRects.size(); i++)
    {
        //for every face rectangle, run landmarkDetector
        full_object_detection landmarks = landmarkDetector(dlibIm, faceRects[i]);

        //print number of landmarks
        if (i==0) cout << "number of landmarks: " << landmarks.num_parts() << endl;

        //store landmarks for current face
        landmarksAll.push_back(landmarks);

        //write landmarks to disk
        std::stringstream landmarksFilename;
        landmarksFilename << landmarksBasename << "_" << i << ".txt";
        cout << "Saving landmarks to " << landmarksFilename.str() << endl;
        writeLandmarksToFile(landmarks, landmarksFilename.str());

        //prepare to draw landmarks and points on face
        Mat imgPoints = img.clone();
        //Read points from eye file
        string pointsFilename = landmarksBasename + "_" + to_string(i) + ".txt";
        //get saved points
        cout << "reading " << pointsFilename << endl;
        const std::vector<cv::Point2f> featurePoints = getSavedPoints(pointsFilename);
        cout << "this many points: " << featurePoints.size() << endl;
        cv::Scalar color = (255,0,0);
        //draw landmarks and points on face
        renderFace(img, landmarks);
        renderFace(imgPoints, featurePoints, color, 10);

        //display images
        int WIDTH = (float)img.cols/5.0;
        int HEIGHT = (float)img.rows/5.0;
        namedWindow("face & lines", WINDOW_NORMAL);
        namedWindow("face & points", WINDOW_NORMAL);
        resizeWindow("face & lines", WIDTH, HEIGHT);
        resizeWindow("face & points", WIDTH, HEIGHT);
        imshow("face & lines", img);
        imshow("face & points", imgPoints);
        int k = waitKey(0);
    }

    return 0;
}
