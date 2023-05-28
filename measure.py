import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt

class ROI(object):

    def __init__(self, x0: int, y0: int, phi: float, length : int, width : int):
        assert(phi >= 0 and phi <= np.pi)
        self.x0 = x0
        self.y0 = y0
        self.phi = phi
        self.length = length
        self.width = width

    def draw_roi(self, img):
        length = self.length
        width = self.width
        phi = self.phi
        x0 = self.x0 - length // 2 * np.cos(phi) - width // 2 * np.cos(phi + np.pi/2)
        y0 = self.y0 - length // 2 * np.sin(phi) - width // 2 * np.sin(phi + np.pi/2)
        x1 = x0 + length * np.cos(phi)
        y1 = y0 + length * np.sin(phi)
        x2 = x1 + width * np.cos(phi + np.pi/2)
        y2 = y1 + width * np.sin(phi + np.pi/2)
        x3 = x0 + width * np.cos(phi + np.pi/2)
        y3 = y0 + width * np.sin(phi + np.pi/2)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)), (0, 0, 0), 2)
    
    def draw_edges(self, img, edges):
        for edge in edges:
            x0, y0 = self.x0, self.y0
            x0 = x0 + edge * np.cos(self.phi) - 5 * np.cos(self.phi + np.pi/2)
            y0 = y0 + edge * np.sin(self.phi) - 5 * np.sin(self.phi + np.pi/2)
            x1 = x0 + 10 * np.cos(self.phi + np.pi/2)
            y1 = y0 + 10 * np.sin(self.phi + np.pi/2)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

    def draw_edge_distances(self, img, edges):
        for i in range(len(edges) - 1):
            x0 = self.x0 + edges[i] * np.cos(self.phi)
            y0 = self.y0 + edges[i] * np.sin(self.phi)
            x1 = self.x0 + edges[i+1] * np.cos(self.phi)
            y1 = self.y0 + edges[i+1] * np.sin(self.phi)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)


def gaussian_filter1d(vec):
    # 1. 高斯滤波
    # 1.1 计算高斯核
    sigma = 1.2
    size = 5
    kernel = np.zeros(size)
    for i in range(size):
        kernel[i] = np.exp(-(i-size//2)**2/(2*sigma**2))
    kernel /= np.sum(kernel)
    # 1.2 高斯滤波
    vec_smoothed = np.zeros(len(vec))
    for i in range(len(vec)):
        for j in range(size):
            if i+j-size//2 >= 0 and i+j-size//2 < len(vec):
                vec_smoothed[i] += vec[i+j-size//2] * kernel[j]
            else:
                vec_smoothed[i] += vec[i] * kernel[j]
    return vec_smoothed


def non_max_suppression(peaks, profile_diff, window_size):
    peaks_refined = []
    for peak in peaks:
        if peak - window_size//2 < 0:
            start = 0
        else:
            start = peak - window_size//2
        if peak + window_size//2 >= len(profile_diff):
            end = len(profile_diff)
        else:
            end = peak + window_size//2
        if np.argmax(profile_diff[start:end]) == peak - start:
            peaks_refined.append(peak)
    return peaks_refined

def measure_1d_rectangle(roi: ROI, img, threshhold=5):
    profile = np.zeros(roi.length+1)
    for i, d in enumerate(range(-roi.length//2, roi.length//2+1)):
        x_p = roi.x0 + d * np.cos(roi.phi)
        y_p = roi.y0 + d * np.sin(roi.phi)

        for w in range(-roi.width//2, roi.width//2+1):
            theta = roi.phi + np.pi // 2
            x = x_p + w * np.cos(theta)
            y = y_p + w * np.sin(theta)
            p = cv2.getRectSubPix(img, (1, 1), (x, y))[0, 0]
            profile[i] += p

    for i in range(roi.length+1):
        profile[i] /= roi.width

    profile_smoothed = gaussian_filter1d(profile)

    profile_diff = np.diff(profile_smoothed)

    # get peaks with abs(value) > threshhold
    peaks = []
    for i in range(1, len(profile_diff)-1):
        if abs(profile_diff[i]) > threshhold and (profile_diff[i] - profile_diff[i-1]) * (profile_diff[i] - profile_diff[i+1]) > 0:
            peaks.append(i)

    # non-max suppression
    peaks = non_max_suppression(peaks, np.abs(profile_diff), window_size=10)
    peaks.sort()

    # refine peak position
    for peak in peaks:
        # given peak - 1, peak, peak + 1, fit a parabola
        x = np.array([peak-1, peak, peak+1])
        y = profile_diff[x]
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # peak position is -b/2a
        peaks[peaks.index(peak)] = -b/2/a - roi.length//2
        print(f"detected edge coordinate in profile line: {-b/2/a - roi.length//2}")

    return peaks

window_size = 100
x0_w, y0_w, w_w, h_w = 670, 1120, window_size, window_size
x1_w, y1_w = 1470, 1120
line_window = 20
line_anchor = [(1097, 1013), (1097, 1142), (1103, 1233), (1119, 1359)]

# 读取图片
img_path = glob.glob('img/*.bmp')
for path in img_path:
    print(f"img: {path}")
    img = cv2.imread(path)
    h, w, _ = img.shape
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi = ROI(1210, 1180, np.pi/2, 450, 22)
    edges = measure_1d_rectangle(roi, gray)
    roi.draw_roi(img)
    roi.draw_edges(img, edges)
    roi.draw_edge_distances(img, edges)

    cv2.namedWindow('line roi', cv2.WINDOW_NORMAL)
    cv2.imshow('line roi', img[roi.y0-350:roi.y0+350, roi.x0-200:roi.x0+200])

    # 1. 圆检测
    # 高斯滤波
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 霍夫圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 100, param1=100, param2=25, minRadius=150, maxRadius=300)

    # 绘制圆
    circle1 = [0, 0, 0]
    circle2 = [0, 0, 0]
    for i in circles[0, :]:
        if i[0] > x0_w and i[0] < x0_w + w_w and i[1] > y0_w and i[1] < y0_w + h_w:
            circle1 = i
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
            cv2.circle(img, (int(i[0]), int(i[1])), 2, (255, 0, 0), 2)
        if i[0] > x1_w and i[0] < x1_w + w_w and i[1] > y1_w and i[1] < y1_w + h_w:
            circle2 = i
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
            cv2.circle(img, (int(i[0]), int(i[1])), 2, (255, 0, 0), 2)
    cv2.rectangle(img, (x0_w, y0_w), (x0_w + w_w, y0_w + h_w), (0, 255, 0), 2)
    cv2.rectangle(img, (x1_w, y1_w), (x1_w + w_w, y1_w + h_w), (0, 255, 0), 2)

    print(f"Hough transform result: r1 = {circle1[2]}, r2 = {circle2[2]}")

    roi_circle1 = ROI(int(circle1[0] + circle1[2] * np.cos(-np.pi * 3/4)), int(circle1[1] + circle1[2] * np.sin(-np.pi* 3/4)), np.pi*1/4, 100, 12)
    roi_circle1.draw_roi(img)
    edges = measure_1d_rectangle(roi_circle1, gray, 5)
    roi_circle1.draw_edges(img, edges)
    roi_circle1.draw_edge_distances(img, edges)
    cv2.namedWindow('circle1 roi', cv2.WINDOW_NORMAL)
    cv2.imshow('circle1 roi', img[roi_circle1.y0-350:roi_circle1.y0+350, roi_circle1.x0-350:roi_circle1.x0+350])

    roi_circle2 = ROI(int(circle2[0] + circle2[2] * np.cos(-np.pi * 1/4)), int(circle2[1] + circle2[2] * np.sin(-np.pi* 1/4)), np.pi*3/4, 100, 12)
    roi_circle2.draw_roi(img)
    edges = measure_1d_rectangle(roi_circle2, gray, 5)
    roi_circle2.draw_edges(img, edges)
    roi_circle2.draw_edge_distances(img, edges)
    cv2.namedWindow('circle2 roi', cv2.WINDOW_NORMAL)
    cv2.imshow('circle2 roi', img[roi_circle2.y0-350:roi_circle2.y0+350, roi_circle2.x0-350:roi_circle2.x0+350])

    # 显示图片
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', lambda event, x, y, flags, param: print(f"x: {x}, y: {y}") if event == cv2.EVENT_LBUTTONDOWN else None)
    cv2.waitKey(0)
    # plt.show()
    cv2.destroyAllWindows()
