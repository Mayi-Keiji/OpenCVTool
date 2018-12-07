#pragma once

#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include "ImageProcess.h"
using namespace cv;
using namespace std;


class CImageDector
{
public:
	CImageDector();
	~CImageDector();
public:
	void CreateMaker(Mat &img, Mat &image);
	void WaterSheld(Mat &img, Mat &out);
	void runWaterSheld(string strPath, Mat &out);
	void runGrabCut(string strPath, Mat &out);
	void runCanny(string strPath, Mat &out);
	void runSobel(string strPath, Mat &out);
	void runLaplas(string strPath, Mat &out);
	void runSobel_Canny(string strPath, Mat &out,int low,int high);
	void run_HoughLine(string strPath, Mat &out, int nMinCount);
	void run_HoughLineP(string strPath, Mat &out, int nMinCount);
	void run_HoughCircle(string strPath, Mat &out, int dp, int para1, int para2);
	void run_Harris(string strPath, Mat &out, int nThread);
	void run_HarrisEx(string strPath, Mat &out, int nThread);
private:
	Mat m_MatMaker;//
	CImageProcess m_IP;
};

