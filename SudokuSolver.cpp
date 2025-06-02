// SUDOKU SOLVER - project
// Andronescu Raluca
// UTCN - CTI en
// Group 30431
// 2025

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <fstream>
#include <fstream>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>
#include <queue>
using namespace std;
using namespace cv;
wchar_t* projectPath;

//-----------------------------------------Sudoku Solver
//
//---------------Blur
bool isInside(Mat img, int i, int j) {
    return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}
Mat_<float> convolution(Mat_<uchar> img, Mat_<float> H) {
    Mat_<float> dst(img.rows, img.cols);
    float s = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            s = 0;
            for (int u = 0; u < H.rows; u++) {
                for (int v = 0; v < H.cols; v++) {
                    int ii = i + u - (H.rows / 2);
                    int jj = j + v - (H.cols / 2);
                    if (isInside(img, ii, jj)) {
                        s += img(ii, jj) * H(u, v);
                    }
                }
            }
            dst(i, j) = s;
        }
    }
    return dst;
}
Mat_<uchar> normalization(Mat_<float> img, Mat_<float> H) {
    Mat_<uchar> dst(img.rows, img.cols);
    float pos = 0, neg = 0;
    float a, b;
    for (int u = 0; u < H.rows; u++) {
        for (int v = 0; v < H.cols; v++) {
            if (H(u, v) > 0)
                pos += H(u, v);
            else
                neg += H(u, v);
        }
    }
    b = pos * 255;
    a = neg * 255;
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            dst(i, j) = (img(i, j) - a) * 255 / (b - a);
    return dst;
}
Mat_< uchar> gaussian_2D(Mat_<uchar> img, int w) {
    Mat_<uchar> result = img.clone();
    float standard_dev = w / 6.0f;
    int half = w / 2;
    double sum = 0;
    float coeff = 1.0f / (2.0f * CV_PI * pow(standard_dev, 2));
    Mat_<float> kernel(w, w);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            float x = i - half;
            float y = j - half;
            kernel(i, j) = coeff * exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(standard_dev, 2)));
        }
    }
    Mat_<float> conv_img = convolution(img, kernel);
    result = normalization(conv_img, kernel);
    //for (int i = 0; i < img.rows; i++) {
    //    for (int j = 0; j < img.cols; j++) {
    //        if (result(i, j) < 128) result(i, j)  =  (0.3* result(i, j));
    //    }
    //}
    return result;
}
//---------------Thresholding
vector<int> calc_hist(Mat_ < uchar> img) {
    vector<int> hist(256, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            hist[img(i, j)]++;
        }
    }
    return hist;
}
vector<float> compute_pdf(Mat_<uchar> img) {
    vector<float> pdf(256, 0.0f);
    int totalPixels = img.rows * img.cols;
    vector<int> hist = calc_hist(img);
    int total = img.rows * img.cols;
    for (int i = 0; i < 256; i++) {
        pdf[i] = (float)hist[i] / total;
    }
    return pdf;
}
Mat_<uchar> multilevel_thresholding(Mat_<uchar> img) {
    vector<int> hist = calc_hist(img);
    vector<float> pdf = compute_pdf(img);

    int WH = 5;
    float TH = 0.005f;

    Mat_<uchar> result = img.clone();
    vector<int> maxima;
    maxima.push_back(0);

    for (int k = WH; k < 256 - WH; k++) {
        float sum = 0;
        for (int i = -WH; i <= WH; i++) {
            sum += pdf[k + i];
        }
        float avg = sum / (2 * WH + 1);
        bool isLocalMax = true;
        for (int i = -WH; i <= WH; i++) {
            if (pdf[k] < pdf[k + i]) {
                isLocalMax = false;
                break;
            }
        }
        if (pdf[k] > avg + TH && isLocalMax) {
            maxima.push_back(k);
            k += WH;
        }
    }
    maxima.push_back(255);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            uchar pixel = img(i, j);
            int closest = maxima[0];
            int minDist = abs(pixel - maxima[0]);

            for (int m = 1; m < maxima.size(); m++) {
                int dist = abs(pixel - maxima[m]);
                if (dist < minDist) {
                    closest = maxima[m];
                    minDist = dist;
                }
            }
            // Invert black and white:
            result(i, j) = 255 - closest;
        }
    }

    return result;
}
Mat_<uchar>  thresholding(Mat_<uchar>& img) {
    int i_min = INT_MAX;
    int i_max = INT_MIN;

    vector<int>  histo = calc_hist(img);

    for (int i = 0; i < histo.size(); i++)
    {
        if (histo[i] > 0)
        {
            i_min = min(i_min, i);
            i_max = max(i_max, i);
        }
    }
    float T = (i_min + i_max) / 2.0;
    float last_T = 0;

    while (1) {
        float m1 = 0, m2 = 0, n1 = 0, n2 = 0;


        for (int i = i_min; i <= i_max; i++)
        {
            if (i < T)
            {
                m1 += (histo[i] * i);
                n1 += histo[i];
            }
            else
            {
                m2 += (histo[i] * i);
                n2 += histo[i];
            }
        }

        m1 = m1 / n1;
        m2 = m2 / n2;
        last_T = T;
        T = (m1 + m2) / 2.0;

        if (abs(T - last_T) <= 0.0) {
            break;
        }
    }

    Mat_<uchar> result(img.size(), uchar(255));
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img(i, j) < T)
            {
                result(i, j) = 0;
            }
            else
            {
                result(i, j) = 255;
            }
        }
    }
    return result;
}
Mat_ <uchar> canny_edge_detection(Mat_<uchar> img) {
    Mat_<float> sk_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat_<float> sk_y = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    Mat_<float> imgx = convolution(img, sk_x);
    Mat_<float> imgy = convolution(img, sk_y); 
    int r = img.rows;
    int c = img.cols;
    Mat_<float> mag(r, c);
    Mat_<float> phi(r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mag(i, j) = sqrt(pow(imgx(i, j), 2) + pow(imgy(i, j), 2));
            phi(i, j) = atan2(imgy(i, j), imgx(i, j));
        }
    }
    Mat_<uchar> dir(r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float ang = phi(i, j);
            if (phi(i, j) <= 0) ang = phi(i, j) + 2 * CV_PI;
            dir(i, j) = (ang * (8 / (2 * CV_PI))) + 0.5;
            dir(i, j) = dir(i, j) % 8;
        }
    }

    int di[] = { 0,-1,-1,-1,0,1,1,1 };
    int dj[] = { 1,1,0,-1,-1,-1,0,1 };

    Mat_<float> mt(r, c);
    mt = mag.clone();
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            int ni = i + di[dir(i, j)];
            int nj = j + dj[dir(i, j)];

            int opi = i - di[dir(i, j)];
            int opj = j - dj[dir(i, j)];

            if (isInside(mt, ni, nj) && mag(ni, nj) > mag(i, j))
                mt(i, j) = 0;
            if (isInside(mt, opi, opj) && mag(opi, opj) > mag(i, j))
                mt(i, j) = 0;
        }
    }
    return mt;


}
Mat_<uchar> edge_linking(Mat_<uchar> mt, int t1, int t2) {
    Mat_<uchar> edges = Mat_<uchar>::zeros(mt.size());
    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            if (mt(i, j) >= t2) {
                edges(i, j) = 255;
            }
            else if (mt(i, j) >= t1) {
                edges(i, j) = 128;
            }
        }
    }

    queue<Point> Q;
    for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
            if (edges(i, j) == 255) {
                Q.push(Point(j, i));
            }
        }
    }

    int dx[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dy[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    while (!Q.empty()) {
        Point p = Q.front();
        Q.pop();

        for (int n = 0; n < 8; n++) {
            int ni = p.y + dy[n];
            int nj = p.x + dx[n];

            if (isInside(mt, ni, nj) && edges(ni, nj) == 128) {
                edges(ni, nj) = 255;
                Q.push(Point(nj, ni));

            }
        }
    }

    for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
            if (edges(i, j) == 128) {
                edges(i, j) = 0;


            }
        }
    }
    // Make image margins black
    int margin = 1;
    for (int i = 0; i < margin; i++) {
        for (int j = 0; j < edges.cols; j++) {
            edges(i, j) = 0;
            edges(edges.rows - 1 - i, j) = 0;
        }
    }
    for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < margin; j++) {
            edges(i, j) = 0;
            edges(i, edges.cols - 1 - j) = 0;
        }
    }
    return edges;


}
Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> str_el) {
    Mat_<uchar> dst = src.clone();

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src(i, j) == 255) {  
                for (int u = 0; u < str_el.rows; u++) {
                    for (int v = 0; v < str_el.cols; v++) {
                        if (str_el(u, v) == 0) {  
                            int new_i = i + u - str_el.rows / 2;
                            int new_j = j + v - str_el.cols / 2;
                            if (isInside(src, new_i, new_j)) {
                                dst(new_i, new_j) = 255;    
                            }
                        }
                    }
                }
            }
        }
    }

    return dst;
}
//---------------Border trace
vector<vector<Point>> borderTrace(const Mat_<uchar>& img) {
    int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
    int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

    vector<vector<Point>> allBorders;
    Mat_<uchar> visited = Mat_<uchar>::zeros(img.size());

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 255 && visited(i, j) == 0) {
                vector<Point> border;

                Point startPoint(j, i);
                Point currentPoint = startPoint;
                border.push_back(currentPoint);
                visited(i, j) = 1;

                int dir = 0;

                bool borderComplete = false;
                while (!borderComplete) {
                    bool foundNext = false;

                    for (int k = 0; k < 8; k++) {
                        int newDir = (dir + k) % 8;

                        int ni = currentPoint.y + di[newDir];
                        int nj = currentPoint.x + dj[newDir];

                        if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols && img(ni, nj) == 255) {
                            currentPoint = Point(nj, ni);
                            border.push_back(currentPoint);
                            visited(ni, nj) = 1;

                            dir = (newDir + 5) % 8;
                            foundNext = true;
                            break;
                        }
                    }
                    if (!foundNext || (border.size() > 2 && currentPoint == startPoint)) {
                        borderComplete = true;
                    }
                }
                if (border.size() > 50) {
                    allBorders.push_back(border);
                }
            }
        }
    }
    return allBorders;
}
vector<vector<Point>> objectPixelsBlack(const Mat_<uchar>& img) {
    int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
    int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

    vector<vector<Point>> allObjects;
    Mat_<uchar> visited = Mat_<uchar>::zeros(img.size());

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0 && visited(i, j) == 0) {
                vector<Point> object;
                queue<Point> q;
                q.push(Point(j, i));
                visited(i, j) = 1;

                while (!q.empty()) {
                    Point p = q.front(); q.pop();
                    object.push_back(p);

                    for (int d = 0; d < 8; d++) {
                        int ni = p.y + di[d];
                        int nj = p.x + dj[d];
                        if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&
                            img(ni, nj) == 0 && visited(ni, nj) == 0) {
                            q.push(Point(nj, ni));
                            visited(ni, nj) = 1;
                        }
                    }
                }
                if (object.size() > 50) { // adjust threshold as needed
                    allObjects.push_back(object);
                }
            }
        }
    }
    return allObjects;
}
Mat drawBorders(const Mat& img, const vector<vector<Point>>& borders) {
    int thickness = 2;
    Mat result = Mat::zeros(img.size(), CV_8UC3);

    if (img.channels() == 3) {
        img.copyTo(result);
    }
    else {
        cvtColor(img, result, COLOR_GRAY2BGR);
    }

    srand(time(NULL));

    for (size_t i = 0; i < borders.size(); i++) {
        uchar r = rand() % 256;
        uchar g = rand() % 256;
        uchar b = rand() % 256;
        Scalar color(b, g, r);
        vector<Point> contour = borders[i];
        for (size_t j = 0; j < contour.size() - 1; j++) {
            line(result, contour[j], contour[j + 1], color, thickness);
        }
        if (contour.size() > 1) {
            line(result, contour[contour.size() - 1], contour[0], color, thickness);
        }
    }
    return result;
}
//---------------Wrap board
vector<Point2f> sortCorners(const vector<Point>& corners) {
    vector<Point2f> ordered(4); 
    vector<int> sum, diff;
    for (auto& pt : corners) {
        sum.push_back(pt.x + pt.y);
        diff.push_back(pt.y - pt.x);
    } 
    auto tlIdx = min_element(sum.begin(), sum.end()) - sum.begin();
    auto brIdx = max_element(sum.begin(), sum.end()) - sum.begin(); 
    auto trIdx = min_element(diff.begin(), diff.end()) - diff.begin(); 
    auto blIdx = max_element(diff.begin(), diff.end()) - diff.begin();
    ordered[0] = corners[tlIdx]; // Top-left
    ordered[1] = corners[trIdx]; // Top-right
    ordered[2] = corners[blIdx]; // Bottom-left
    ordered[3] = corners[brIdx]; // Bottom-right
    return ordered;
}
Mat warpSudokuBoard(const Mat& inputImage, const vector<Point>& corners, Size outputSize = Size(800, 800)) {
    if (corners.size() != 4) {
        cerr << "Error: corners vector must contain exactly 4 points." << endl;
        return inputImage.clone();
    }
    vector<Point2f> ord = sortCorners(corners);
    Point2f src[4];
    for (int i = 0; i < 4; i++) {
        src[i] = ord[i];
    } 

    Point2f dst[4] = {
        Point2f(0, 0),
        Point2f(outputSize.width, 0),
        Point2f(0, outputSize.height),
        Point2f(outputSize.width, outputSize.height)
    };

    Mat matrix = getPerspectiveTransform(src, dst);

    Mat warped;
    warpPerspective(inputImage, warped, matrix, outputSize);

    return warped;
}
//---------------Localize
Mat_<uchar> localizeSudoku(Mat& sudokuImage)
{
    Mat gray;
    cvtColor(sudokuImage, gray, COLOR_BGR2GRAY);
    imshow("Gray Image", gray);
    waitKey(0); 
 
    //-----Blur & Treshhold & Edge detection
     Mat_<uchar> edges = canny_edge_detection(gray);
    imshow("Canny Edges", edges);
    waitKey(0); 
    Mat_<uchar> linked = edge_linking(edges, 100, 170);
    imshow("Linked Edges", linked);
    waitKey(0); 
    uchar data[] = {
        0,   0,   0,
        0,   0,   0,
        0,   0,   0
    };

    //-----Dilation to make the borders bigger
    Mat_<uchar> str_el(3, 3, data);
    Mat_<uchar> dil = dilation(linked, str_el);
    imshow("Image Dilation", dil);
    waitKey(0); 

    //-----Contours 
    vector<vector<Point>> contours = borderTrace(dil);
    Mat contourImage2 = drawBorders(sudokuImage, contours);
    imshow("All Contours", contourImage2);
    waitKey(0); 

    //Find biggest contour
    double maxArea = 0;
    vector<Point> biggest;
    for (auto& contour : contours)
    {
        double area = contourArea(contour);
        if (area > maxArea)
        {
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * arcLength(contour, true), true);
            if (approx.size() == 4)
            {
                biggest = approx;
                maxArea = area;
            }
        }
    }

    if (biggest.size() == 4)
    {
        vector<vector<Point>> drawBiggest = { biggest };
        Mat boardOutline = drawBorders(sudokuImage, drawBiggest);
        imshow("Biggest Contour", boardOutline);
        waitKey(0); 
    }


    if (biggest.size() != 4)
    {
        printf("No Sudoku grid detected.\n");
        return sudokuImage;
    }

    Point2f src[4]; 

    sort(biggest.begin(), biggest.end(), [](Point a, Point b) { return a.y < b.y; });
    if (biggest[0].x < biggest[1].x)
    {
        src[0] = biggest[0];
        src[1] = biggest[1];
    }
    else
    {
        src[0] = biggest[1];
        src[1] = biggest[0];
    }

    if (biggest[2].x < biggest[3].x)
    {
        src[2] = biggest[2];
        src[3] = biggest[3];
    }
    else
    {
        src[2] = biggest[3];
        src[3] = biggest[2];
    }
     
    Mat warped = warpSudokuBoard(sudokuImage, biggest);
    
    imshow("Warped Sudoku Board", warped);
    waitKey(0); 

    return warped;
}
//---------------Cells
Mat_<uchar> findCellBorder(const Mat_<uchar>& cell, int borderWidth  ) {
    Mat_<uchar> cellNoBorder = cell.clone();
    int rows = cellNoBorder.rows, cols = cellNoBorder.cols;
    for (int y = 0; y < borderWidth; ++y)
        for (int x = 0; x < cols; ++x) {
            cellNoBorder(y, x) = 255;                         // Top
            cellNoBorder(rows - 1 - y, x) = 255;              // Bottom
        }
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < borderWidth; ++x) {
            cellNoBorder(y, x) = 255;                         // Left
            cellNoBorder(y, cols - 1 - x) = 255;              // Right
        }
    return cellNoBorder;
}
Mat_<uchar> removeBorderObjects(const Mat_<uchar>& img) {
    Mat_<uchar> result(img.rows, img.cols);
    result.setTo(255);

    vector<vector<Point>> objects = objectPixelsBlack(img);
    if (objects.empty()) {
        return result;  
    }
    Mat_<uchar> noBorder  (img.rows, img.cols);
    noBorder.setTo(0);
    Mat_<uchar> noBorder2 = findCellBorder(noBorder, 5);


    for (const auto& obj : objects) {
        bool touchesBorder = false;
        for (const auto& pt : obj) {
            if (noBorder2(pt.y, pt.x) == 255) {
                touchesBorder = true;
                break;
            }
        }
        if (!touchesBorder) {
            for (const auto& pt : obj) {
                result(pt.y, pt.x) = 0;
            }
        }
    }
    return result;
}
//---------------Digit Recognition
Mat_<uchar> centerAndScaleDigit(const Mat_<uchar>& cell, Size targetSize = Size(98, 100)) {
    Mat_<uchar> bin;
    threshold(cell, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
 
    int pad = max(cell.rows, cell.cols) / 10;  
    Mat_<uchar> padded(bin.rows + 2 * pad, bin.cols + 2 * pad, uchar(255));
    bin.copyTo(padded(Rect(pad, pad, bin.cols, bin.rows)));
     
    Mat inv = 255 - padded;
    vector<Point> pts;
    findNonZero(inv, pts);
    if (pts.empty())
        return Mat_<uchar>(targetSize, uchar(255));
    Rect bbox = boundingRect(pts);
    Mat digitCrop = padded(bbox);
     
    int targetW = int(targetSize.width * 0.8);
    int targetH = int(targetSize.height * 0.8);
    double scale = min(double(targetW) / digitCrop.cols, double(targetH) / digitCrop.rows);
    int newW = max(1, int(digitCrop.cols * scale));
    int newH = max(1, int(digitCrop.rows * scale));

    Mat digitResized;
    resize(digitCrop, digitResized, Size(newW, newH), 0, 0, INTER_AREA);
     
    Mat_<uchar> output(targetSize, uchar(255));
    int x = (targetSize.width - newW) / 2;
    int y = (targetSize.height - newH) / 2;
    digitResized.copyTo(output(Rect(x, y, newW, newH)));

    return output;
}
double pixelMatchScore(const Mat_<uchar>& centered, const Mat_<uchar>& tmpl,int i) {
    int matchCount = 0, cellBlack = 0, tmplBlack = 0;
    for (int y = 0; y < centered.rows; ++y)
        for (int x = 0; x < centered.cols; ++x) {
            if (centered(y, x) == 0 && tmpl(y, x) == 0) matchCount++;
            if (centered(y, x) == 0  ) cellBlack++;
            if ( tmpl(y, x) == 0) tmplBlack++;
        }
    return 2.0 * matchCount / (cellBlack + tmplBlack);
}
//---------------Grid
vector<Mat_<uchar>> splitSudokuGrid(const Mat_<uchar>& sudokuBoard, Size templateSize = Size(98, 100)) {
    vector<Mat_<uchar>> cells;
    int cellHeight = sudokuBoard.rows / 9;
    int cellWidth = sudokuBoard.cols / 9;
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            Rect cellRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
            Mat_<uchar> cell = sudokuBoard(cellRect).clone();
            Mat_<uchar> resizedCell;
            resize(cell, resizedCell, templateSize, 0, 0, INTER_AREA);
            cells.push_back(resizedCell);
        }
    }
    return cells;
}
Mat_<int> recognizeSudokuGrid(const vector<Mat_<uchar>>& cells, const vector<Mat_<uchar>>& digitTemplates) {
    Mat_<int> grid(9, 9); grid.setTo(0);
    for (int idx = 0; idx < cells.size(); ++idx) {
        int row = idx / 9, col = idx % 9;
        Mat_<uchar> cell = cells[idx].clone();
        //Thresholding the cell
        
        Mat_<uchar> thCell = thresholding(cell);
    
        //Resize
        Mat_<uchar> resizedCell;
        resize(thCell, resizedCell, digitTemplates[0].size());
        //Remove borgers and center digit
        Mat_<uchar> removedBorder = removeBorderObjects(resizedCell);
        Mat_<uchar> centered = centerAndScaleDigit(removedBorder); 
        //Check if cell is empty
        double blackRatio = (centered.total() - countNonZero(centered)) / double(centered.total());
        if (blackRatio < 0.07) { 
            grid(row, col) = 0; continue; 
        }    
       /* imshow("im1", cell); 
        imshow("im2",thCell);
        imshow("im3", removedBorder);
        imshow("im4", centered);
        waitKey(0);*/
        //Digit Recognition
        int bestDigit = 0;
        double bestScore = 0.0;
        for (int i = 0; i < digitTemplates.size(); ++i) {
            double score = pixelMatchScore(centered, digitTemplates[i],i); 
            if (score > bestScore) {
                bestScore = score;

                bestDigit = i % 9 +1;
            }
        }        
        if (bestScore < 0.01)  
            grid(row, col) = 0;
        else
            grid(row, col) = bestDigit;
    }
     return grid;
}
//---------------Solve
bool isValid(const Mat_<int>& grid, int row, int col, int num) {
    for (int i = 0; i < 9; i++) {
        if (grid(row, i) == num || grid(i, col) == num)
            return false;
    }
    int boxRow = row - row % 3, boxCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (grid(boxRow + i, boxCol + j) == num)
                return false;
    return true;
}
bool solveSudokuHelper(Mat_<int>& grid) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (grid(row, col) == 0) {
                for (int num = 1; num <= 9; num++) {
                    if (isValid(grid, row, col, num)) {
                        grid(row, col) = num;
                        if (solveSudokuHelper(grid))
                            return true;
                        grid(row, col) = 0;
                    }
                }
                return false; 
            }
        }
    }
    return true;  
}
Mat_<int> solveSudoku(const Mat_<int>& inputGrid) {
    Mat_<int> grid = inputGrid.clone();
    if (solveSudokuHelper(grid))
        return grid;  
    else
        return inputGrid;  
}
//---------------Print
void printSudokuMatrix(const Mat_<int>& grid) {
    for (int i = 0; i < grid.rows; i++) {
        for (int j = 0; j < grid.cols; j++) {
            cout << grid(i, j) << " ";
            if ((j + 1) % 3 == 0 && j != grid.cols - 1)
                cout << "| ";
        }
        cout << endl;
        if ((i + 1) % 3 == 0 && i != grid.rows - 1)
            cout << "------+-------+------" << endl;
    }
}
Mat drawSudokuMatrix(const Mat_<int>& grid,int ok, int cellSize = 50) {
    int n = 9;
    int imgSize = n * cellSize + 1;
    Mat sudokuImg(imgSize, imgSize, CV_8UC3, Scalar(255, 255, 255));

    //Draw grid lines
    for (int i = 0; i <= n; i++) {
        int thickness = (i % 3 == 0) ? 3 : 1;
        line(sudokuImg, Point(0, i * cellSize), Point(imgSize, i * cellSize), Scalar(0, 0, 0), thickness);
        line(sudokuImg, Point(i * cellSize, 0), Point(i * cellSize, imgSize), Scalar(0, 0, 0), thickness);
    } 
    //Draw numbers
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int num = grid(i, j);
            if (num > 0) {
                string text = to_string(num);
                int font = FONT_HERSHEY_SIMPLEX;
                double fontScale = 1.0;
                int thickness = 2;
                int baseline = 0;
                Size textSize = getTextSize(text, font, fontScale, thickness, &baseline);
                Point textOrg(j * cellSize + (cellSize - textSize.width) / 2,
                    i * cellSize + (cellSize + textSize.height) / 2);
                putText(sudokuImg, text, textOrg, font, fontScale, Scalar(0, 0, 0), thickness);
            }
        }
    }
    if (ok == 1) {
        imshow("Solved Sudoku", sudokuImg);
    }
    else {
        imshow("Unsolved Sudoku", sudokuImg);
    }
    waitKey(0);
    return sudokuImg;
}
//---------------Load  
vector<Mat_<uchar>> loadDigitTemplates() {
    _wchdir(projectPath);
    _wchdir(L"Digits");
    vector<Mat_<uchar>> templates;
    for (int digit = 1; digit <=135; digit++) {
        string filename = "nr_" + to_string(digit) + ".png";
        Mat_<uchar> img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Could not load " << filename << endl;
            continue;
        } 
        templates.push_back(img);
    }
    //My data set conytains 135 pictures
    return templates;
}
void loadSudokuImage(Mat& sudokuImage, vector<Mat_<uchar>> digitTemplates)
{
    _wchdir(projectPath);
    _wchdir(L"Images");

    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        sudokuImage = imread(fname, IMREAD_COLOR);
        if (sudokuImage.empty())
        {
            printf("Could not open or find the Sudoku image!\n");
            continue;
        }
        resize(sudokuImage, sudokuImage, Size(600, 600));
        imshow("Loaded Sudoku Image", sudokuImage);
        waitKey(0); 
        Mat_<uchar> wrappedSudoku = localizeSudoku(sudokuImage);
        vector<Mat_<uchar>> cells = splitSudokuGrid(wrappedSudoku);
        Mat_<int> grid = recognizeSudokuGrid(cells, digitTemplates);
        cout << " Unsolved Sudoku " << endl;
        printSudokuMatrix(grid);
        drawSudokuMatrix(grid,0);
        waitKey(0);
        Mat_<int> solved = solveSudoku(grid);
        cout << " Solved Sudoku " << endl;
        printSudokuMatrix(solved);
        drawSudokuMatrix(solved,1);
        waitKey(0);
        break;
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    vector<Mat_<uchar>> digitTemplates = loadDigitTemplates();

    int op;
    do
    {
        system("cls");
        destroyAllWindows();
        printf("Sudoku solver:\n");
        printf("1. Load Sudoku Image\n");
        printf("0. Exit\n");
        printf("Option: ");
        scanf("%d", &op);

        if (op == 1)
        {
            Mat sudokuImage;

            loadSudokuImage(sudokuImage, digitTemplates);

        }

    } while (op != 0);

    return 0;
}
