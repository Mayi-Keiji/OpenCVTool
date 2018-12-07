#include "stdafx.h"
#include "DrawItem.h"
// OpenCV Headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
// C++ Standard Libraries
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std;

/************************************************************************/
/* 定义与地理栅格数据测试相关的变量                                       */
/************************************************************************/
// define the corner points
//    Note that GDAL library can natively determine this
cv::Point2d tl(-122.441017, 37.815664);
cv::Point2d tr(-122.370919, 37.815311);
cv::Point2d bl(-122.441533, 37.747167);
cv::Point2d br(-122.3715, 37.746814);
// determine dem corners
cv::Point2d dem_bl(-122.0, 38);
cv::Point2d dem_tr(-123.0, 37);
// range of the heat map colors
std::vector<std::pair<cv::Vec3b, double> > color_range;

// List of all function prototypes
cv::Point2d lerp(const cv::Point2d&, const cv::Point2d&, const double&);
cv::Vec3b get_dem_color(const double&);
cv::Point2d world2dem(const cv::Point2d&, const cv::Size&);
cv::Point2d pixel2world(const int&, const int&, const cv::Size&);
void add_color(cv::Vec3b& pix, const uchar& b, const uchar& g, const uchar& r);

/*
* Linear Interpolation
* p1 - Point 1
* p2 - Point 2
* t  - Ratio from Point 1 to Point 2
*/
cv::Point2d lerp(cv::Point2d const& p1, cv::Point2d const& p2, const double& t) {
	return cv::Point2d(((1 - t)*p1.x) + (t*p2.x),
		((1 - t)*p1.y) + (t*p2.y));
}
/*
* Interpolate Colors
*/
template <typename DATATYPE, int N>
cv::Vec<DATATYPE, N> lerp(cv::Vec<DATATYPE, N> const& minColor,
	cv::Vec<DATATYPE, N> const& maxColor,
	double const& t) {
	cv::Vec<DATATYPE, N> output;
	for (int i = 0; i<N; i++) {
		output[i] = (uchar)(((1 - t)*minColor[i]) + (t * maxColor[i]));
	}
	return output;
}
/*
* Compute the dem color
*/
cv::Vec3b get_dem_color(const double& elevation) {
	// if the elevation is below the minimum, return the minimum
	if (elevation < color_range[0].second) {
		return color_range[0].first;
	}
	// if the elevation is above the maximum, return the maximum
	if (elevation > color_range.back().second) {
		return color_range.back().first;
	}
	// otherwise, find the proper starting index
	int idx = 0;
	double t = 0;
	for (int x = 0; x<(int)(color_range.size() - 1); x++) {
		// if the current elevation is below the next item, then use the current
		// two colors as our range
		if (elevation < color_range[x + 1].second) {
			idx = x;
			t = (color_range[x + 1].second - elevation) /
				(color_range[x + 1].second - color_range[x].second);
			break;
		}
	}
	// interpolate the color
	return lerp(color_range[idx].first, color_range[idx + 1].first, t);
}
/*
* Given a pixel coordinate and the size of the input image, compute the pixel location
* on the DEM image.
*/
cv::Point2d world2dem(cv::Point2d const& coordinate, const cv::Size& dem_size) {
	// relate this to the dem points
	// ASSUMING THAT DEM DATA IS ORTHORECTIFIED
	double demRatioX = ((dem_tr.x - coordinate.x) / (dem_tr.x - dem_bl.x));
	double demRatioY = 1 - ((dem_tr.y - coordinate.y) / (dem_tr.y - dem_bl.y));
	cv::Point2d output;
	output.x = demRatioX * dem_size.width;
	output.y = demRatioY * dem_size.height;
	return output;
}
/*
* Convert a pixel coordinate to world coordinates
*/
cv::Point2d pixel2world(const int& x, const int& y, const cv::Size& size) {
	// compute the ratio of the pixel location to its dimension
	double rx = (double)x / size.width;
	double ry = (double)y / size.height;
	// compute LERP of each coordinate
	cv::Point2d rightSide = lerp(tr, br, ry);
	cv::Point2d leftSide = lerp(tl, bl, ry);
	// compute the actual Lat/Lon coordinate of the interpolated coordinate
	return lerp(leftSide, rightSide, rx);
}
/*
* Add color to a specific pixel color value
*/
void add_color(cv::Vec3b& pix, const uchar& b, const uchar& g, const uchar& r) {
	if (pix[0] + b < 255 && pix[0] + b >= 0) { pix[0] += b; }
	if (pix[1] + g < 255 && pix[1] + g >= 0) { pix[1] += g; }
	if (pix[2] + r < 255 && pix[2] + r >= 0) { pix[2] += r; }
}


const int NUMBER = 100;
const int DELAY = 50;

const int window_width = 900;
const int window_height = 600;
int x_1 = -window_width / 2;
int x_2 = window_width * 3 / 2;
int y_1 = -window_width / 2;
int y_2 = window_width * 3 / 2;


CDrawItem::CDrawItem()
{
	strcpy_s(window_name, "Drawing Demo");
	rng = RNG(0xFFFFFFFF);
	image = Mat::zeros(window_height, window_width, CV_8UC3);
}


CDrawItem::~CDrawItem()
{
}


void CDrawItem::DrawLine( char* name)
{
	Point pt1, pt2;
	for (int i = 0;i < NUMBER; i++)
	{
		pt1.x = rng.uniform(x_1, x_2);
		pt1.y = rng.uniform(y_1, y_2);
		pt2.x = rng.uniform(x_1, x_2);
		pt2.y = rng.uniform(y_1, y_2);
		/*
		//! type of line
		enum LineTypes {
		FILLED  = -1,
		LINE_4  = 4, //!< 4-connected line
		LINE_8  = 8, //!< 8-connected line
		LINE_AA = 16 //!< antialiased line
		};
		*/
		int nLineType = 0;
		if (i %2 == 0)
		{
			nLineType = 4;
		}
		if (i %3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		line(image, pt1, pt2, RandomColor(rng), rng.uniform(1, 10), nLineType);
		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return ;
		}
	}
	return;
}
void CDrawItem::DrawRectangles( char* name )
{
	Point pt1, pt2;
	int nLineType = 0;
	int nThichness = rng.uniform(-3, 10);
	for (int i = 0; i < NUMBER; i++)
	{
		pt1.x = rng.uniform(x_1, x_2);
		pt1.y = rng.uniform(y_1, y_2);
		pt2.x = rng.uniform(x_1, x_2);
		pt2.y = rng.uniform(y_1, y_2);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		rectangle(image, pt1, pt2, RandomColor(rng), MAX(nThichness, -1), nLineType);
		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
}
void CDrawItem::DrawEllipses( char* name )
{
	int nLineType = 0;
	for (int i=0;i < NUMBER; i ++)
	{
		Point center;
		center.x = rng.uniform(x_1, x_2);
		center.y = rng.uniform(x_1, x_2);

		Size axes;
		axes.height = rng.uniform(0, 200);
		axes.width = rng.uniform(0, 200);

		double angle = rng.uniform(0, 100);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		/*
			img:图像。
			center:椭圆圆心坐标。
			axes :轴的长度。
			angle:偏转的角度。
			start_angle :圆弧起始角的角度。
			end_angle :圆弧终结角的角度。
			color :线条的颜色。
			thickness :线条的粗细程度。
			line_type :线条的类型,见CVLINE的描述。
			shift :圆心坐标点和数轴的精度。
		*/
		ellipse(image, center, axes, angle, angle - 100, angle + 200, RandomColor(rng), rng.uniform(-1, 9), nLineType);
		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
	return;
}
void CDrawItem::DrawPolyLines( char* name )
{
	int nLineType = 8;

	for (int i = 0; i < NUMBER; i++)
	{
		Point pt[2][3];
		pt[0][0].x = rng.uniform(x_1, x_2);
		pt[0][0].y = rng.uniform(y_1, y_2);
		pt[0][1].x = rng.uniform(x_1, x_2);
		pt[0][1].y = rng.uniform(y_1, y_2);
		pt[0][2].x = rng.uniform(x_1, x_2);
		pt[0][2].y = rng.uniform(y_1, y_2);
		pt[1][0].x = rng.uniform(x_1, x_2);
		pt[1][0].y = rng.uniform(y_1, y_2);
		pt[1][1].x = rng.uniform(x_1, x_2);
		pt[1][1].y = rng.uniform(y_1, y_2);
		pt[1][2].x = rng.uniform(x_1, x_2);
		pt[1][2].y = rng.uniform(y_1, y_2);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		const Point* ppt[2] = { pt[0], pt[1] };
		int npt[] = { 3, 3 };
		//怎么用？ 
		/*
			img	：作为画布的矩阵
			pts	：折线顶点数组
			npts	：折线顶点个数
			ncontours	：待绘制折线数
			isClosed	：是否是闭合折线(多边形)
			color	：折线的颜色
			thickness	：折线粗细
			lineType	：线段类型
		*/
		polylines(image, ppt, npt, 2, true, RandomColor(rng), rng.uniform(1, 10), nLineType);
		
		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
	return ;
	
}
void CDrawItem::DrawFilledPolyLines( char* name )
{
	int nLineType = 8;

	for (int i = 0; i < NUMBER; i++)
	{
		Point pt[2][3];
		pt[0][0].x = rng.uniform(x_1, x_2);
		pt[0][0].y = rng.uniform(y_1, y_2);
		pt[0][1].x = rng.uniform(x_1, x_2);
		pt[0][1].y = rng.uniform(y_1, y_2);
		pt[0][2].x = rng.uniform(x_1, x_2);
		pt[0][2].y = rng.uniform(y_1, y_2);
		pt[1][0].x = rng.uniform(x_1, x_2);
		pt[1][0].y = rng.uniform(y_1, y_2);
		pt[1][1].x = rng.uniform(x_1, x_2);
		pt[1][1].y = rng.uniform(y_1, y_2);
		pt[1][2].x = rng.uniform(x_1, x_2);
		pt[1][2].y = rng.uniform(y_1, y_2);

		const Point* ppt[2] = { pt[0], pt[1] };
		int npt[] = { 3, 3 };
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		fillPoly(image, ppt, npt, 2, RandomColor(rng), nLineType);

		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
	return ;
}
void CDrawItem::DrawCircles( char* name )
{
	int nLineType = 0;
	for (int i=0;i < NUMBER; i++)
	{
		Point center;
		center.x = rng.uniform(x_1, x_2);
		center.y = rng.uniform(x_1, x_2);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		circle(image, center, rng.uniform(0, 300), RandomColor(rng), rng.uniform(-1, 9), nLineType);

		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return ;
		}
	}
	return;
}
void CDrawItem::DrawTests( char* name )
{
	int nLineType = 0;
	for (int i=0;i < NUMBER; i++)
	{
		Point org;
		org.x = rng.uniform(x_1, x_2);
		org.y = rng.uniform(x_1, x_2);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}
		putText(image, "Testing Test Render", org, rng.uniform(0, 8),
			rng.uniform(0, 100)*0.05 + 0.1, RandomColor(rng), rng.uniform(1, 10), nLineType
		);

		imshow(name, image);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
}
void CDrawItem::DrawBigEnd( char* name )
{
	int nLineType = 0;
	Size textSize = getTextSize("OpenCV forever!", FONT_HERSHEY_COMPLEX, 3, 5, 0);
	Point org((window_width - textSize.width) / 2, (window_height - textSize.height) / 2);

	Mat image2;
	for (int i = 0;i < 255; i+=2)
	{
		image2 = image - Scalar::all(i);
		if (i % 2 == 0)
		{
			nLineType = 4;
		}
		if (i % 3 == 0)
		{
			nLineType = 8;
		}
		if (i % 5 == 0)
		{
			nLineType = 16;
		}

		putText(image2, "OpenCV forever!", org, FONT_HERSHEY_COMPLEX, 3, Scalar(i, i, 255), 5, nLineType);
		imshow(name, image2);
		if (waitKey(DELAY) >= 0)
		{
			return;
		}
	}
	return;
}

Scalar CDrawItem::RandomColor(RNG &rng)
{
	int iColor = (unsigned)rng;
	return Scalar(iColor & 255, (iColor >> 8) & 255, (iColor >> 16) & 255);
}
//操作地理空间栅格数据
void CDrawItem::Geospatial()
{

	// load the image (note that we don't have the projection information.  You will
	// need to load that yourself or use the full GDAL driver.  The values are pre-defined
	// at the top of this file
	cv::Mat image = cv::imread("D:/gdal_output.jpg", cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR);
	// load the dem model
	cv::Mat dem = cv::imread("D:/N37W123.hgt", cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);
	// create our output products
	cv::Mat output_dem(image.size(), CV_8UC3);
	cv::Mat output_dem_flood(image.size(), CV_8UC3);
	// for sanity sake, make sure GDAL Loads it as a signed short
	//if (dem.type() != CV_16SC1) { throw std::runtime_error("DEM image type must be CV_16SC1"); }
	// define the color range to create our output DEM heat map
	//  Pair format ( Color, elevation );  Push from low to high
	//  Note:  This would be perfect for a configuration file, but is here for a working demo.
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(188, 154, 46), -1));
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(110, 220, 110), 0.25));
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(150, 250, 230), 20));
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(160, 220, 200), 75));
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(220, 190, 170), 100));
	color_range.push_back(std::pair<cv::Vec3b, double>(cv::Vec3b(250, 180, 140), 200));
	// define a minimum elevation
	double minElevation = -10;
	// iterate over each pixel in the image, computing the dem point
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			// convert the pixel coordinate to lat/lon coordinates
			cv::Point2d coordinate = pixel2world(x, y, image.size());
			// compute the dem image pixel coordinate from lat/lon
			cv::Point2d dem_coordinate = world2dem(coordinate, dem.size());
			// extract the elevation
			double dz;
			if (dem_coordinate.x >= 0 && dem_coordinate.y >= 0 &&
				dem_coordinate.x < dem.cols && dem_coordinate.y < dem.rows) {
				dz = dem.at<short>(dem_coordinate);
			}
			else {
				dz = minElevation;
			}
			// write the pixel value to the file
			output_dem_flood.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
			// compute the color for the heat map output
			cv::Vec3b actualColor = get_dem_color(dz);
			output_dem.at<cv::Vec3b>(y, x) = actualColor;
			// show effect of a 10 meter increase in ocean levels
			if (dz < 10) {
				add_color(output_dem_flood.at<cv::Vec3b>(y, x), 90, 0, 0);
			}
			// show effect of a 50 meter increase in ocean levels
			else if (dz < 50) {
				add_color(output_dem_flood.at<cv::Vec3b>(y, x), 0, 90, 0);
			}
			// show effect of a 100 meter increase in ocean levels
			else if (dz < 100) {
				add_color(output_dem_flood.at<cv::Vec3b>(y, x), 0, 0, 90);
			}
		}
	}
	// print our heat map
	cv::imwrite("heat-map.jpg", image);
	// print the flooding effect image
	cv::imwrite("flooded.jpg", output_dem_flood);
	imshow("source", output_dem);
	waitKey(0);
	imshow("heat_map", output_dem);
	waitKey(0);
	imshow("output_dem_flood", output_dem_flood);
	waitKey(0);

}
