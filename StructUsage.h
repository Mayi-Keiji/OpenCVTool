#pragma once

#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

class CStructUsage
{
public:
	CStructUsage();
	~CStructUsage();
	void ScalarUsage();
	void MatUsage();
	void RNG_Usage();
	void MakeBorder_Usage(Mat &img);
	void Parallelize();
};

