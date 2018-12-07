#pragma once


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;

class CDrawItem
{
public:
	CDrawItem();
	~CDrawItem();
	void DrawLine(  char* name );
	void DrawRectangles(  char* name );
	void DrawEllipses(  char* name );
	void DrawPolyLines(  char* name );
	void DrawFilledPolyLines(  char* name );
	void DrawCircles(  char* name );
	void DrawTests(  char* name );
	void DrawBigEnd(  char* name );
	Scalar RandomColor(RNG &rng);
	//∂¡–¥µÿ¿ÌÕº
	void Geospatial();
private:
	char window_name[256];
	RNG rng;
	Mat image;

};

