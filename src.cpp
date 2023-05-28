#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <vector>

using namespace cv;
using namespace std;

const double pi = 3.1415926535;

struct idpoint
{
    double x;
    double y;
};

struct roi
{
    int x;
    int y;
    int h;
    int w;
};

vector<Point2f> oneD_edge_horizontal(Mat src, Rect roi, int interval, int mean_num)
{
    vector<Point2f> point_find;
    Mat img = src.clone();
    double point_value[175] = { 0 };
    double point_diff[175] = { 0 };
    double point_diff_max = 0;
    double point_location = 0;
    idpoint point_max = { 0 };
    idpoint point_2 = { 0 };
    double src_value = 0;
    int point_num = 0;
    int n = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    point_num = roi.height / interval;
    for (i = 0; i < point_num; i++)
    {
        for (j = 0; j < roi.width; j++)
        {
            src_value = src.at<uchar>(roi.y + i * interval, roi.x + j);
            point_value[j] = src_value;
        }
        for (j = 0; j < roi.width - 1; j++)
        {
            point_diff[j + 1] = abs(point_value[j + 1] - point_value[j]);
            if (abs(point_diff[j + 1]) >= point_diff_max)
            {
                point_diff_max = point_diff[j + 1];
                point_max.y = roi.y + i * interval;
                point_max.x = roi.x + j + 1;
            }
            point_diff[j + 1] = 0;
        }
        point_location = point_max.x;
        point_2.x = point_location;
        point_2.y = roi.y + i * interval;
        Point2f point_find_infor(point_2.x, point_2.y);
        point_find.push_back(point_find_infor);
        point_diff_max = 0;
    }
    return point_find;
}
Point houghCircleDetection_1(vector<Point2f>& point_find, int minRadius, int maxRadius)
{
    int angle = 0;
    int ax[2000];
    int ay[2000];
    memset(ax, 0, sizeof(ax));
    memset(ay, 0, sizeof(ay));
    int xmax = 0;
    int ymax = 0;
    int i = 0;
    int x = 0;
    int y = 0;
    Point center;
    int radius = 0;
    for (i = 0; i < point_find.size(); i++)
    {
        for (radius = minRadius; radius < maxRadius; radius++)
        {
            for (angle = 0; angle < 360; angle++)
            {
                x = point_find[i].x + radius * cos(angle * pi / 180);
                y = point_find[i].y + radius * sin(angle * pi / 180);
                if (x >= 0 && x < 2000)
                {
                    if (y >= 0 && y < 2000)
                    {
                        ax[x]++;
                        ay[y]++;
                    }
                }
            }
        }
    }

    for (x = 0; x < 2000; x++)
    {
        for (y = 0; y < 2000; y++)
        {
            if (ax[x] >= xmax && ay[y] >= ymax)
            {
                if (y >= 1000 && x >= 500)
                {
                    center.x = x;
                    center.y = y;
                    xmax = ax[x];
                    ymax = ay[y];
                }

            }
        }
    }
    return center;
}

double draw(vector<Point2f>& point_find, Mat src, Point op)
{
    int i = 0;
    double rnum = 0;
    int rx = 0;
    int ry = 0;
    double r = 0;
    for (i = 0; i < point_find.size(); i++)
    {
        circle(src, Point(point_find[i].x, point_find[i].y), 3, 255, 3);
    }
    for (i = 0; i < point_find.size(); i++)
    {
        rx = abs(point_find[i].x - op.x);
        ry = abs(point_find[i].y - op.y);
        r = sqrtf(rx * rx + ry * ry);
        rnum = rnum + r;
    }
    rnum = rnum / point_find.size();
    circle(src, Point(op.x, op.y), int(rnum), 255, 3);
    return rnum;
}


int main(int argc, char** argv)
{
    const char* filename = "img/1.bmp";

    Mat img = imread(filename, 0);
    Mat imgc = imread(filename);
    double r1, r2;
    if (img.empty())
    {
        cout << "can not open " << filename << endl;
        return -1;
    }

    Mat img2, img3;
    GaussianBlur(img, img2, Size(13, 13), 2, 2);

    //threshold(img2, img3,195,255, THRESH_BINARY);
    //namedWindow("t", 0);
    //imshow("t", img3);

    Rect roi_r_left = Rect(500, 1050, 145, 300);
    vector<Point2f> point_find_r_left;
    point_find_r_left = oneD_edge_horizontal(img2, roi_r_left, 1, 20);
    Point center1 = houghCircleDetection_1(point_find_r_left, 120, 140);
    circle(imgc, center1, 3, 255, 3);
    r1 = draw(point_find_r_left, imgc, center1);

    Rect roi_r_right = Rect(1590, 1020, 150, 300);
    vector<Point2f> point_find_r_right;
    point_find_r_right = oneD_edge_horizontal(img2, roi_r_right, 1, 20);
    Point center2 = houghCircleDetection_1(point_find_r_right, 120, 140);
    circle(imgc, center2, 3, 255, 3);
    r2 = draw(point_find_r_right, imgc, center2);

    rectangle(imgc, Point(1590, 1020), Point(1740, 1320), Scalar(0, 255, 255), 5);
    rectangle(imgc, Point(500, 1050), Point(645, 1350), Scalar(0, 255, 255), 5);


    namedWindow("s", 0);
    imshow("s", imgc);
    cout << "center1" << center1 << endl;
    cout << "r1=" << r1 << endl;
    cout << "center2" << center2 << endl;
    cout << "r2=" << r2 << endl;



    waitKey();



    return 0;
}

