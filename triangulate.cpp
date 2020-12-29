#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

using namespace cv;
using namespace std;

//given a vector of points, find the index of the point closest to the input point.
static int 
findIndex(vector<Point2f>& points, Point2f &point)
{
    int minIndex = 0;
    double minDistance = norm(points[0] - point);
    for(int i = 1; i < points.size(); i++)
    {
        double distance = norm(points[i] - point);
        if(distance < minDistance)
        {
            minIndex = i;
            minDistance = distance;
        }
    }
    return minIndex;
}

//write delaunay triangles to file
static void 
writeDelaunay(Subdiv2D& subdiv, vector<Point2f>& points, const string &filename)
{
    //Open file for writing
    std::ofstream ofs;
    ofs.open(filename);

    //get list of triangles
    //each triangle is stored as a vector of 6 coordinates
    //(x0, y0, x1, y1, x2, y2)
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    //convert triangle representation to three vertices
    vector<Point2f> vertices(3);

    //loop over all triangles and write to file
    for(size_t i = 0; i < triangleList.size(); i++)
    {
        //current triangle
        Vec6f t = triangleList[i];

        //get vertices of current triangle
        vertices[0] = Point2f(t[0], t[1]);
        vertices[1] = Point2f(t[2], t[3]);
        vertices[2] = Point2f(t[4], t[5]);

        //find indices of vertices in the points list
        int landmark1 = findIndex(points, vertices[0]);
        int landmark2 = findIndex(points, vertices[1]);
        int landmark3 = findIndex(points, vertices[2]);

        //save to file
        ofs << landmark1 << " " << landmark2 << " " << landmark2 << endl;
    }
    ofs.close();
}

static void
drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunayColor)
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    //convert triangle representation to three vertices
    vector<Point> vertices(3);

    //get size of image
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);

    for(size_t i = 0; i < triangleList.size(); i++)
    {
        //current triangle 
        Vec6f t = triangleList[i];

        //convert triangle to vertices
        vertices[0] = Point(cvRound(t[0]), cvRound(t[1]));
        vertices[1] = Point(cvRound(t[2]), cvRound(t[3]));
        vertices[2] = Point(cvRound(t[4]), cvRound(t[5]));

        //draw triangles that are completely inside the image
        if(rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
        {
            line(img, vertices[0], vertices[1], delaunayColor, 5, LINE_AA, 0);
            line(img, vertices[1], vertices[2], delaunayColor, 5, LINE_AA, 0);
            line(img, vertices[2], vertices[0], delaunayColor, 5, LINE_AA, 0);
        }
    }
}

static void
drawPoint(Mat& img, Point2f fp, Scalar color)
{
    circle(img, fp, 10, color, FILLED, LINE_AA, 0);
}

int
main()
{
    //create a vector of points
    vector<Point2f> points;

    //read in points from a text file
    string pointsFilename("./images/reference_face_0.txt");
    ifstream ifs(pointsFilename);
    int x, y;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x, y));
    }

    cout << "Reading file " << pointsFilename << endl;

    //find bounding box enclosing the points
    Rect rect = boundingRect(points);
    //create instance of Subdiv2D
    Subdiv2D subdiv(rect);
    //insert points into subdiv 
    for(vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }

    //output filename
    string outputFileName("./images/reference_face_0.tri");
    //write delaunay triangles
    writeDelaunay(subdiv, points, outputFileName);

    cout << "Writing Delaunay triangles to " << outputFileName << endl;

    //image for displaying Delaunay Triangulation
    //read image
    string imageFilename("./images/reference_face.png");
    cv::Mat img = cv::imread(imageFilename);
    Mat imgDelaunay;
    imgDelaunay = img.clone();
    //define colors for drawing
    Scalar delaunayColor(255,255,255), pointsColor(0,0,255);
    //draw deLaunay triangles
    drawDelaunay(imgDelaunay, subdiv, delaunayColor);
    //draw points
    for(vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        drawPoint(imgDelaunay, *it, pointsColor);
    }

    //display images
    int WIDTH = (float)imgDelaunay.cols/5.0;
    int HEIGHT = (float)imgDelaunay.rows/5.0;
    namedWindow("delaunay triangles", WINDOW_NORMAL);
    resizeWindow("delaunay triangles", WIDTH, HEIGHT);
    imshow("delaunay triangles", imgDelaunay);
    int k = waitKey(0);

    return 0;
}

