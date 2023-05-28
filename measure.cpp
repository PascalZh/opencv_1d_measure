#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>
using namespace std;
using namespace cv;

struct ROI {
  ROI(int x0, int y0, float phi, int length, int width) {
    assert(phi >= 0 && phi <= M_PI);
    this->x0 = x0;
    this->y0 = y0;
    this->phi = phi;
    this->length = length;
    this->width = width;
  }

  void draw_roi(Mat& img) {
    int length = this->length;
    int width = this->width;
    float phi = this->phi;
    int x0_ = this->x0 - float(length / 2) * cos(phi) - float(width / 2) * cos(phi + M_PI / 2);
    int y0_ = this->y0 - float(length / 2) * sin(phi) - float(width / 2) * sin(phi + M_PI / 2);
    int x1 = x0_ + length * cos(phi);
    int y1 = y0_ + length * sin(phi);
    int x2 = x1 + width * cos(phi + M_PI / 2);
    int y2 = y1 + width * sin(phi + M_PI / 2);
    int x3 = x0_ + width * cos(phi + M_PI / 2);
    int y3 = y0_ + width * sin(phi + M_PI / 2);
    line(img, Point(x0_, y0_), Point(x1, y1), Scalar(0, 0, 0), 2);
    line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 0), 2);
    line(img, Point(x2, y2), Point(x3, y3), Scalar(0, 0, 0), 2);
    line(img, Point(x3, y3), Point(x0_, y0_), Scalar(0, 0, 0), 2);
  }

  void draw_edges(Mat& img, vector<int>& edges) {
    for (int edge : edges) {
      int x1 = this->x0, y1 = this->y0;
      x1 = x0 + edge * cos(this->phi) - 5 * cos(this->phi + M_PI / 2);
      y1 = y0 + edge * sin(this->phi) - 5 * sin(this->phi + M_PI / 2);
      int x2 = x1 + 10 * cos(this->phi + M_PI / 2);
      int y2 = y1 + 10 * sin(this->phi + M_PI / 2);
      line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 1);
    }
  }

  void draw_edge_distances(Mat& img, vector<int>& edges) {
    if (edges.size() < 2) return;
    for (int i = 0; i < edges.size() - 1; i++) {
      int x1 = this->x0 + edges[i] * cos(this->phi);
      int y1 = this->y0 + edges[i] * sin(this->phi);
      int x2 = this->x0 + edges[i + 1] * cos(this->phi);
      int y2 = this->y0 + edges[i + 1] * sin(this->phi);
      line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 1);
    }
  }

  int x0, y0;
  float phi;
  int length, width;
};

vector<double> gaussian_filter1d(vector<double>& vec) {
  // 1. 高斯滤波
  // 1.1 计算高斯核
  double sigma = 1.2;
  int size = 5;
  vector<double> kernel(size);
  for (int i = 0; i < size; i++) {
    kernel[i] = exp(-pow(i - size / 2, 2) / (2 * pow(sigma, 2)));
  }
  double sum = accumulate(kernel.begin(), kernel.end(), 0.0);
  for (double& v : kernel) v /= sum;
  // 1.2 高斯滤波
  vector<double> vec_smoothed(vec.size());
  for (int i = 0; i < vec.size(); i++) {
    for (int j = 0; j < size; j++) {
      if (i + j - size / 2 >= 0 && i + j - size / 2 < vec.size()) {
        vec_smoothed[i] += vec[i + j - size / 2] * kernel[j];
      } else {
        vec_smoothed[i] += vec[i] * kernel[j];
      }
    }
  }
  return vec_smoothed;
}

vector<int> non_max_suppression(vector<int>& peaks,
                                vector<double>& profile_diff, int window_size) {
  int start = 0, end = 0;
  vector<int> peaks_refined;
  for (int peak : peaks) {
    if (peak - window_size / 2 < 0) {
      start = 0;
    } else {
      start = peak - window_size / 2;
    }
    if (peak + window_size / 2 >= profile_diff.size()) {
      end = profile_diff.size();
    } else {
      end = peak + window_size / 2;
    }
    if (max_element(profile_diff.begin() + start, profile_diff.begin() + end) -
            profile_diff.begin() ==
        peak - start) {
      peaks_refined.push_back(peak);
    }
  }
  return peaks_refined;
}

vector<int> measure_1d_rectangle(ROI& roi, Mat& img, int threshhold = 5) {
  vector<double> profile(roi.length + 1);
  for (int i = 0; i <= roi.length; i++) {
    double d = i - roi.length / 2;
    int x_p = roi.x0 + d * cos(roi.phi);
    int y_p = roi.y0 + d * sin(roi.phi);
    for (int w = -roi.width / 2; w <= roi.width / 2; w++) {
      double theta = roi.phi + M_PI / 2;
      int x = x_p + w * cos(theta);
      int y = y_p + w * sin(theta);
      double p = img.at<uchar>(y, x);
      profile[i] += p;
    }
    profile[i] /= roi.width;
  }
  vector<double> profile_smoothed = gaussian_filter1d(profile);
  vector<double> profile_diff(profile_smoothed.size());
  adjacent_difference(profile_smoothed.begin(), profile_smoothed.end(),
                      profile_diff.begin());
  // get peaks with abs(value) > threshhold
  vector<int> peaks;
  for (int i = 1; i < profile_diff.size() - 1; i++) {
    if (abs(profile_diff[i]) > threshhold &&
        (profile_diff[i] - profile_diff[i - 1]) *
        (profile_diff[i] - profile_diff[i + 1]) > 0) {
      peaks.push_back(i);
    }
  }
  // peaks = non_max_suppression(peaks, profile_diff, 10);
  sort(peaks.begin(), peaks.end());

  cout << "peak nums: " << peaks.size() << endl;
  float peak_float;
  for (int& peak : peaks) {
    int x[3] = {peak - 1, peak, peak + 1};
    double y[3] = {profile_diff[x[0]], profile_diff[x[1]], profile_diff[x[2]]};
    Mat A(3, 3, CV_64F);
    A.at<double>(0, 0) = x[0] * x[0];
    A.at<double>(0, 1) = x[0];
    A.at<double>(0, 2) = 1;
    A.at<double>(1, 0) = x[1] * x[1];
    A.at<double>(1, 1) = x[1];
    A.at<double>(1, 2) = 1;
    A.at<double>(2, 0) = x[2] * x[2];
    A.at<double>(2, 1) = x[2];
    A.at<double>(2, 2) = 1;
    Mat a(3, 1, CV_64F), b(3, 1, CV_64F);
    b.at<double>(0) = y[0];
    b.at<double>(1) = y[1];
    b.at<double>(2) = y[2];
    solve(A, b, a);
    peak_float = -a.at<double>(1) / 2 / a.at<double>(0) - roi.length / 2;
    peak = int(peak_float);
    cout << peak_float << endl;
  }
  return peaks;
}

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

    ROI roi(1210, 1180, M_PI / 2, 450, 22);
    vector<int> edges = measure_1d_rectangle(roi, gray);
    roi.draw_roi(img);
    roi.draw_edges(img, edges);
    roi.draw_edge_distances(img, edges);

    namedWindow("line roi", WINDOW_NORMAL);
    imshow("line roi", img(Rect(roi.x0 - 350, roi.y0 - 350, 700, 700)));

    GaussianBlur(gray, gray, Size(3, 3), 0);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 2, 100, 100, 25, 150, 300);

    Vec3f circle1, circle2;
    for (size_t i = 0; i < circles.size(); i++) {
      Vec3f c = circles[i];
      if (c[0] > x0_w && c[0] < x0_w + w_w && c[1] > y0_w &&
          c[1] < y0_w + h_w) {
        circle1 = c;
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]),
               Scalar(0, 0, 255), 2);
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), 2, Scalar(255, 0, 0),
               2);
      }
      if (c[0] > x1_w && c[0] < x1_w + w_w && c[1] > y1_w &&
          c[1] < y1_w + h_w) {
        circle2 = c;
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]),
               Scalar(0, 0, 255), 2);
        circle(img, Point(cvRound(c[0]), cvRound(c[1])), 2, Scalar(255, 0, 0),
               2);
      }
    }
    rectangle(img, Point(x0_w, y0_w), Point(x0_w + w_w, y0_w + h_w),
              Scalar(0, 255, 0), 2);
    rectangle(img, Point(x1_w, y1_w), Point(x1_w + w_w, y1_w + h_w),
              Scalar(0, 255, 0), 2);

    cout << "Hough transform result: r1 = " << circles[0][2]
         << ", r2 = " << circles[1][2] << endl;

    ROI roi_circle1(int(circle1[0] + circle1[2] * cos(-M_PI * 3 / 4)),
                    int(circle1[1] + circle1[2] * sin(-M_PI * 3 / 4)),
                    M_PI / 4, 100, 12);
    roi_circle1.draw_roi(img);
    edges = measure_1d_rectangle(roi_circle1, gray, 5);
    roi_circle1.draw_edges(img, edges);
    roi_circle1.draw_edge_distances(img, edges);
    namedWindow("circle1 roi", WINDOW_NORMAL);
    imshow("circle1 roi",
           img(Rect(roi_circle1.x0 - 350, roi_circle1.y0 - 350, 700, 700)));

    ROI roi_circle2(int(circle2[0] + circle2[2] * cos(-M_PI / 4)),
                    int(circle2[1] + circle2[2] * sin(-M_PI / 4)),
                    M_PI * 3 / 4, 100, 12);
    roi_circle2.draw_roi(img);
    edges = measure_1d_rectangle(roi_circle2, gray, 5);
    roi_circle2.draw_edges(img, edges);
    roi_circle2.draw_edge_distances(img, edges);
    namedWindow("circle2 roi", WINDOW_NORMAL);
    imshow("circle2 roi",
           img(Rect(roi_circle2.x0 - 350, roi_circle2.y0 - 350, 700, 700)));

    imshow("img", img);
    setMouseCallback("img",
                     [](int event, int x, int y, int flags, void* param) {
                       if (event == EVENT_LBUTTONDOWN)
                         cout << "x: " << x << ", y: " << y << endl;
                     });
    waitKey(0);
  }
}
