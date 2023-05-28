#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main() {
  int window_size = 100;
  int x0_w = 670, y0_w = 1120, w_w = window_size, h_w = window_size;
  int x1_w = 1470, y1_w = 1120;
  int line_window = 20;
  vector<Point2f> line_anchor = {Point2f(1097, 1013), Point2f(1097, 1142),
                                 Point2f(1103, 1233), Point2f(1119, 1359)};

  vector<String> img_path;
  glob("img/*.bmp", img_path);
  for (const auto& path : img_path) {
    cout << "img: " << path << endl;
    Mat img = imread(path);
    int h = img.rows;
    int w = img.cols;
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gray, (Size(3, 3)), 0);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 2, 100, 100, 25, 150, 300);
    Mat edges;
    Canny(gray, edges, 30, 90, 3);
    namedWindow("edges", WINDOW_NORMAL);
    imshow("edges", edges);
    vector<Vec2f> lines;
    HoughLines(edges, lines, 0.5, CV_PI / 180 / 5, 180);

    vector<Vec2f> lines_new;
    int i;
    lines_new.push_back(Vec2f(0, 0));
    lines_new.push_back(Vec2f(0, 0));
    lines_new.push_back(Vec2f(0, 0));
    lines_new.push_back(Vec2f(0, 0));
    i = 0;
    for (const auto& anchor : line_anchor) {
      int cnt = 0;
      for (const auto& l : lines) {
        float rho = l[0];
        float theta = l[1];
        float dist = abs(anchor.x * cos(theta) + anchor.y * sin(theta) - rho);
        if (dist < line_window) {
          lines_new[i] += l;
          cnt++;
        }
      }
      if (cnt != 0) {
        lines_new[i] /= cnt;
      }
      i++;
    }
    float d1 =
        abs(lines_new[0][0] -
            lines_new[3][0] * (cos(lines_new[3][1]) * cos(lines_new[0][1]) +
                               sin(lines_new[3][1]) * sin(lines_new[0][1])));
    float d2 =
        abs(lines_new[1][0] -
            lines_new[2][0] * (cos(lines_new[2][1]) * cos(lines_new[1][1]) +
                               sin(lines_new[2][1]) * sin(lines_new[1][1])));
    cout << "d1 = " << d1 << ", d2 = " << d2 << endl;

    for (const auto& l : lines_new) {
      float rho = l[0];
      float theta = l[1];
      float a = cos(theta);
      float b = sin(theta);
      float x = a * rho;
      float y = b * rho;
      int x0 = cvRound(x + h * (-b));
      int y0 = cvRound(y + h * (a));
      int x1 = cvRound(x - h * (-b));
      int y1 = cvRound(y - h * (a));
      line(img, Point(x0, y0), Point(x1, y1), Scalar(0, 0, 255));
    }

    Vec3f circle1, circle2;
    int cnt1, cnt2;
    for (size_t i = 0; i < circles.size(); i++) {
      Vec3f c = circles[i];
      if (c[0] > x0_w && c[0] < x0_w + w_w && c[1] > y0_w &&
          c[1] < y0_w + h_w) {
        circle1 += c;
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]),
               Scalar(0, 0, 255), 2);
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), 2, Scalar(255, 0, 0),
               2);
        cnt1++;
      }
      if (c[0] > x1_w && c[0] < x1_w + w_w && c[1] > y1_w &&
          c[1] < y1_w + h_w) {
        circle2 += c;
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]),
               Scalar(0, 0, 255), 2);
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), 2, Scalar(255, 0, 0),
               2);
        cnt2++;
      }
    }
    if (cnt1 != 0) {
      circle1 /= cnt1;
    }
    if (cnt2 != 0) {
      circle2 /= cnt2;
    }
    rectangle(img, Point(x0_w, y0_w), Point(x0_w + w_w, y0_w + h_w),
              Scalar(0, 255, 0), 2);
    rectangle(img, Point(x1_w, y1_w), Point(x1_w + w_w, y1_w + h_w),
              Scalar(0, 255, 0), 2);

    float r1 = circle1[2];
    float r2 = circle2[2];
    cout << "r1 = " << r1 << ", r2 = " << r2 << endl;
    namedWindow("img", WINDOW_NORMAL);
    imshow("img", img);
    setMouseCallback(
        "img",
        [](int event, int x, int y, int flags, void* param) {
          if (event == EVENT_LBUTTONDOWN)
            cout << "x: " << x << ", y: " << y << endl;
        },
        NULL);
    waitKey(0);
    destroyAllWindows();
  }

  return 0;
}
