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
using namespace std;
using namespace cv;
wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
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
//---------------Load image
Mat localizeSudoku(Mat& sudokuImage)
{
    Mat gray;
    cvtColor(sudokuImage, gray, COLOR_BGR2GRAY);
    imshow("Gray Image", gray);
    waitKey(0);

    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    imshow("Blurred Image", blurred);
    waitKey(0);

    Mat thresh;
    adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
    imshow("Threshold Image", thresh);
    waitKey(0);

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = sudokuImage.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 2);
    imshow("All Contours", contourImage);
    waitKey(0);

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

    Mat boardOutline = sudokuImage.clone();
    if (biggest.size() == 4)
    {
        vector<vector<Point>> drawBiggest = { biggest };
        drawContours(boardOutline, drawBiggest, -1, Scalar(0, 0, 255), 5);
    }
    imshow("Biggest Contour", boardOutline);
    waitKey(0);

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
    Mat warped;
    warpPerspective(sudokuImage, warped, matrix, Size(800, 800));

    imshow("Warped Sudoku Board", warped);
    waitKey(0);

    return warped;
}

void loadSudokuImage(Mat& sudokuImage)
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
        localizeSudoku(sudokuImage);
        break;
    }
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

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
			loadSudokuImage(sudokuImage);
           
		}

	} while (op != 0);

	return 0;
}
