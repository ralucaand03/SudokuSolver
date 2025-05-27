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

using namespace std;
using namespace cv;
wchar_t* projectPath;

void testOpenImage()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src;
        src = imread(fname);
        imshow("image", src);
        waitKey();
    }
}

void testOpenImagesFld()
{
    char folderName[MAX_PATH];
    if (openFolderDlg(folderName) == 0)
        return;
    char fname[MAX_PATH];
    FileGetter fg(folderName, "bmp");
    while (fg.getNextAbsFile(fname))
    {
        Mat src;
        src = imread(fname);
        imshow(fg.getFoundFileName(), src);
        if (waitKey() == 27) //ESC pressed
            break;
    }
}

void testImageOpenAndSave()
{
    _wchdir(projectPath);

    Mat src, dst;

    src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

    if (!src.data)	// Check for invalid input
    {
        printf("Could not open or find the image\n");
        return;
    }

    // Get the image resolution
    Size src_size = Size(src.cols, src.rows);

    // Display window
    const char* WIN_SRC = "Src"; //window for the source image
    namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
    moveWindow(WIN_SRC, 0, 0);

    const char* WIN_DST = "Dst"; //window for the destination (processed) image
    namedWindow(WIN_DST, WINDOW_AUTOSIZE);
    moveWindow(WIN_DST, src_size.width + 10, 0);

    cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

    imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

    imshow(WIN_SRC, src);
    imshow(WIN_DST, dst);

    waitKey(0);
}
//-----------------------------------Sudoku Solver
// extrag cells si compar cu o poza de referinta 
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
                result(i, j) = 255;
            }
            else
            {
                result(i, j) = 0;
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
    //imshow("IMG", img);
    //imshow("IMG  x", abs(imgx) / 255);
    //imshow("IMG  y", abs(imgy) / 255);
    //waitKey(0);
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
            if (src(i, j) == 255) {  // WHITE = foreground now
                for (int u = 0; u < str_el.rows; u++) {
                    for (int v = 0; v < str_el.cols; v++) {
                        if (str_el(u, v) == 0) {  // structuring element active
                            int new_i = i + u - str_el.rows / 2;
                            int new_j = j + v - str_el.cols / 2;
                            if (isInside(src, new_i, new_j)) {
                                dst(new_i, new_j) = 255;  // spread WHITE
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
Mat warpSudokuBoard(const Mat& inputImage, const vector<Point>& corners, Size outputSize = Size(800, 800)) {
    if (corners.size() != 4) {
        cerr << "Error: corners vector must contain exactly 4 points." << endl;
        return inputImage.clone();
    }

    Point2f src[4];
    for (int i = 0; i < 4; i++) {
        src[i] = corners[i];
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
   /* imshow("Gray Image", gray);
    waitKey(0);*/

    //-----Blur  
    //Mat_<uchar> blurred = gaussian_2D(gray, 4);
    //imshow("Blurred Image", blurred);
    //waitKey(0);

    //-----Treshhold  
    /*Mat_<uchar> thresh = thresholding(blurred);
    imshow("Threshold Image", thresh);
    waitKey(0);*/
    Mat_<uchar> thresh = canny_edge_detection(gray);
    /*imshow("Threshold Image", thresh);
    waitKey(0);*/
    Mat_<uchar> thresh2 = edge_linking(thresh, 100, 170);
   /* imshow("Threshold Image", thresh2);
    waitKey(0);*/
    uchar data[] = {
        0,   0,   0,
        0,   0,   0,
        0,   0,   0
    };
    Mat_<uchar> str_el(3, 3, data);
    Mat_<uchar> dil = dilation(thresh2, str_el);
   /* imshow("Threshold Image Dilatation", dil);
    waitKey(0);*/
    //-----Contours 
    vector<vector<Point>> contours = borderTrace(dil);
    Mat contourImage2 = drawBorders(sudokuImage, contours);
   /* imshow("All Contours", contourImage2);
    waitKey(0);*/

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
       /* imshow("Biggest Contour", boardOutline);
        waitKey(0);*/
    }


    if (biggest.size() != 4)
    {
        printf("No Sudoku grid detected.\n");
        return sudokuImage;
    }

    Point2f src[4];
    Point2f dst[4];

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

    dst[0] = Point2f(0, 0);
    dst[1] = Point2f(800, 0);
    dst[2] = Point2f(0, 800);
    dst[3] = Point2f(800, 800);

    Mat matrix = getPerspectiveTransform(src, dst);
    Mat warped = warpSudokuBoard(sudokuImage, biggest);
    /*imshow("Warped Sudoku Board", warped);
    waitKey(0);*/

    return warped;
}
//---------------Cells
void showSudokuCells(const Mat_<uchar>& sudokuBoard) {
    int cellHeight = sudokuBoard.rows / 9;
    int cellWidth = sudokuBoard.cols / 9;

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            // Extract cell using simple row/col slicing
            Mat_<uchar> cell = sudokuBoard(Range(i * cellHeight, (i + 1) * cellHeight),
                Range(j * cellWidth, (j + 1) * cellWidth));

            // Show the cell
            imshow("Cell " + to_string(i) + "," + to_string(j), cell);
            waitKey(100); // small pause to show windows properly
        }
    }
}
vector<Mat_<uchar>> loadDigitTemplates() {
    _wchdir(projectPath);
    _wchdir(L"Digits");
    vector<Mat_<uchar>> templates;
    for (int digit = 1; digit <= 9; digit++) {
        string filename = "nr_" + to_string(digit) + ".png";
        Mat_<uchar> img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Could not load " << filename << endl;
            continue;
        }
        /*imshow("Digit " + to_string(digit)  , img);
        waitKey(100);*/
        templates.push_back(img);
    }
    return templates;
}
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

//int recognizeDigit(const Mat_<uchar>& cell, const vector<Mat_<uchar>>& digitTemplates) {
//    // Preprocess the cell for better matching
//    Mat_<uchar> processedCell;
//    threshold(cell, processedCell, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
//
//    // Remove noise (optional, but improves results)
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
//    morphologyEx(processedCell, processedCell, MORPH_OPEN, kernel);
//
//    int bestDigit = 0;
//    double bestScore = -1.0;
//
//    for (int i = 0; i < digitTemplates.size(); ++i) {
//        // Resize cell digit to template size
//        Mat_<uchar> resizedCell;
//        resize(processedCell, resizedCell, digitTemplates[i].size());
//
//        // Compare using matchTemplate
//        Mat result;
//        matchTemplate(resizedCell, digitTemplates[i], result, TM_CCOEFF_NORMED);
//
//        double minVal, maxVal;
//        Point minLoc, maxLoc;
//        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
//
//        if (maxVal > bestScore) {
//            bestScore = maxVal;
//            bestDigit = i + 1; // digits are 1-indexed
//        }
//    }
//
//    // Optional: Set a threshold for confidence, e.g., 0.4
//    if (bestScore < 0.4) return 0; // Consider cell empty if confidence is too low
//
//    return bestDigit;
//}
//Mat_<int> recognizeSudokuGrid(const Mat_<uchar>& sudokuBoard, const vector<Mat_<uchar>>& digitTemplates) {
//    Mat_<int> grid(9, 9); grid.setTo(0);
//    int cellHeight = sudokuBoard.rows / 9;
//    int cellWidth = sudokuBoard.cols / 9;
//    for (int row = 0; row < 9; row++) {
//        for (int col = 0; col < 9; col++) {
//            Rect cellRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
//            Mat_<uchar> cell = sudokuBoard(cellRect).clone();
//            Mat_<uchar> processedCell;
//            threshold(cell, processedCell, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
//            double blackRatio = (processedCell.total() - countNonZero(processedCell)) / double(cellWidth * cellHeight);
//            if (blackRatio < 0.08) { grid(row, col) = 0; continue; }
//            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
//            morphologyEx(processedCell, processedCell, MORPH_OPEN, kernel);
//            int bestDigit = 0;
//            double bestDiff = 1e9;
//            for (int i = 0; i < digitTemplates.size(); ++i) {
//                Mat_<uchar> tmpl = digitTemplates[i].clone();
//                Mat_<uchar> resizedCell, resizedTemplate;
//                resize(processedCell, resizedCell, tmpl.size());
//                threshold(tmpl, resizedTemplate, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
//                int cellBlack = resizedCell.total() - countNonZero(resizedCell);
//                int tmplBlack = resizedTemplate.total() - countNonZero(resizedTemplate);
//                int cellMid = resizedCell.rows / 2;
//                int tmplMid = resizedTemplate.rows / 2;
//                int cellBlackTop = resizedCell(Range(0, cellMid), Range::all()).total() - countNonZero(resizedCell(Range(0, cellMid), Range::all()));
//                int cellBlackBottom = resizedCell(Range(cellMid, resizedCell.rows), Range::all()).total() - countNonZero(resizedCell(Range(cellMid, resizedCell.rows), Range::all()));
//                int tmplBlackTop = resizedTemplate(Range(0, tmplMid), Range::all()).total() - countNonZero(resizedTemplate(Range(0, tmplMid), Range::all()));
//                int tmplBlackBottom = resizedTemplate(Range(tmplMid, resizedTemplate.rows), Range::all()).total() - countNonZero(resizedTemplate(Range(tmplMid, resizedTemplate.rows), Range::all()));
//                double diff = abs(cellBlack - tmplBlack)
//                    + abs(cellBlackTop - tmplBlackTop)
//                    + abs(cellBlackBottom - tmplBlackBottom);
//                if (diff < bestDiff) { bestDiff = diff; bestDigit = i + 1; }
//            }
//            if (bestDiff > 0.5 * cellWidth * cellHeight) grid(row, col) = 0;
//            else grid(row, col) = bestDigit;
//        }
//    }
//    return grid;
//}
Mat_<uchar> removeCellBorder(const Mat_<uchar>& cell, int borderWidth = 2) {
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
double pixelMatchScore(const Mat_<uchar>& cell, const Mat_<uchar>& tmpl) {
    Mat_<uchar> resizedCell;
    resize(cell, resizedCell, tmpl.size()); 
    Mat_<uchar> removedBorder = removeCellBorder(resizedCell,14); 
    int matchCount = 0, total = cell.rows * cell.cols;
    for (int y = 0; y < removedBorder.rows; ++y)
        for (int x = 0; x < removedBorder.cols; ++x)
            if (removedBorder(y, x) == 0 && tmpl(y, x) == 0) matchCount++; 
    return double(matchCount);  
}
Mat_<int> recognizeSudokuGrid(const vector<Mat_<uchar>>& cells, const vector<Mat_<uchar>>& digitTemplates) {
    Mat_<int> grid(9, 9); grid.setTo(0);
    for (int idx = 0; idx < cells.size(); ++idx) {
        int row = idx / 9, col = idx % 9;
        Mat_<uchar> cell = cells[idx].clone();

        // Crop borders: remove 1 pixel from each edge (more if borders are thicker)
        int border = 1;
        if (cell.rows > 2 * border && cell.cols > 2 * border)
            cell = cell(Range(border, cell.rows - border), Range(border, cell.cols - border)).clone();

        // Threshold and check for emptiness
        Mat_<uchar> thCell;
        threshold(cell, thCell, 0, 255, THRESH_BINARY | THRESH_OTSU);

        /*imshow("cell", thCell);
        waitKey(0);*/
        double blackRatio = (thCell.total() - countNonZero(thCell)) / double(thCell.total());
        if (blackRatio < 0.07) { grid(row, col) = 0; continue; }

        int bestDigit = 0;
        double bestScore = 0.0;
        for (int i = 0; i < digitTemplates.size(); ++i) {
            double score = pixelMatchScore(thCell, digitTemplates[i]);
            if (score > bestScore) {
                bestScore = score;
                bestDigit = i + 1;
            }
        }
        // Only accept confident matches
        if (bestScore <  20)  // adjust threshold if needed
            grid(row, col) = 0;
        else
            grid(row, col) = bestDigit;
    }
    return grid;
}

void showFirstNineCells(const vector<Mat_<uchar>>& cells) {
    int n = min(9, (int)cells.size());
    for (int i = 0; i < n; ++i) {
        imshow("Cell " + to_string(i), cells[i]);
        waitKey(0);  // Wait for key press before showing next
    }
    destroyAllWindows();
}
//---------------Load image
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
        //cout << "------+-------+------" << endl;
        Mat_<uchar> wrappedSudoku = localizeSudoku(sudokuImage);
        vector<Mat_<uchar>> cells = splitSudokuGrid(wrappedSudoku);
        //showFirstNineCells(cells);
        Mat_<int> grid = recognizeSudokuGrid(cells, digitTemplates);
        printSudokuMatrix(grid);
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
