#pragma once
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
enum Filter_Type
{
	TYPE_MEAN,
	TYPE_GUSSIAN,
	TYPE_MEDIAN,
	TYPE_bILATERAL
};

class CImageProcess
{
public:
	CImageProcess();
	~CImageProcess();
	bool LoadImage(int nMode, string path);
	bool SaveImage(int nMode, string path);
	Mat& GetImage();
	bool GetGaryImage(Mat &img, Mat &gray);
	bool TraverseImage(Mat &img);
	bool TraverseImage1(Mat &img);
	bool ThreadHold(int nThread, Mat &gray, Mat &res, int nType);
	bool Enrode(Mat &img, Mat &out, int nType, int nSize);
	bool Dilate(Mat &img, Mat &out, int nType, int nSize);
	bool Close(Mat &img, Mat &out, int nType, int nSize);
	bool Open(Mat &img, Mat &out, int nType, int nSize);
	bool Gradient(Mat &img, Mat &out, int nType, int nSize);
	bool TopHat(Mat &img, Mat &out, int nType, int nSize);
	bool BlackHat(Mat &img, Mat &out, int nType, int nSize);
	bool HitOrMiss(Mat &img, Mat &out, int nType, int nSize);
	bool CreateKernel(int nSize, int ar[]);
	bool Blur(Mat &img, Mat &out, Filter_Type eType, int nSize, float fSigmaX = 0.0f, float fSigmaY = 0.0f);
	bool BlurH(Mat &img, Mat&out, int nSize);
	bool BlurV(Mat &img, Mat&out, int nSize);
	bool Filter(Mat &img, Mat&out, Mat& kernel);
	bool ContrastBrightness(Mat &img, Mat &out, float alpha, float beta);
	bool AddTwoImage(Mat &img1, Mat&img2, Mat&out, float fAlpha, int x = 0, int y = 0);
	bool Fourier(Mat &img, Mat &out);
private:
	Mat m_Img;
};

