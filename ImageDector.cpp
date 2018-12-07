#include "stdafx.h"
#include "ImageDector.h"
#include "harrisDetector.h"
#define  PI 360

Vec3b RandomColor(int value)
{
	value = value % 255;  //生成0~255的随机数
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}
CImageDector::CImageDector()
{
}


CImageDector::~CImageDector()
{
}
void CImageDector::CreateMaker(Mat &imageGray,Mat &image)
{
	//首先对图像做预处理
	GaussianBlur(imageGray, imageGray, Size(5, 5), 2);   //高斯滤波
	imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 80, 150);
	imshow("Canny Image", imageGray);

	//查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //轮廓	
	Mat marks(image.size(), CV_32S);   //Opencv分水岭第二个矩阵参数
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	imshow("marksShow", marksShows);
	imshow("轮廓", imageContours);
	m_MatMaker = marks.clone();
}
void CImageDector::WaterSheld(Mat &img, Mat &out)
{
	cv::watershed(img, m_MatMaker);
	out = m_MatMaker.clone();
}

void CImageDector::runWaterSheld(string strPath, Mat &out)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);
	CreateMaker(imageGray, image);
	watershed(image, m_MatMaker);

	//我们再来看一下分水岭算法之后的矩阵marks里是什么东西
	Mat afterWatershed;
	convertScaleAbs(m_MatMaker, afterWatershed);
	imshow("After Watershed", afterWatershed);

	//对每一个区域进行颜色填充
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < m_MatMaker.rows; i++)
	{
		for (int j = 0; j < m_MatMaker.cols; j++)
		{
			int index = m_MatMaker.at<int>(i, j);
			if (m_MatMaker.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	imshow("After ColorFill", PerspectiveImage);

	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	imshow("AddWeighted Image", wshed);
	out = wshed.clone();
}

void CImageDector::runGrabCut(string strPath, Mat &out)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);
	cv::Rect rectangle(50, 70, image.cols - 150, image.rows - 180);
	cv::Mat result;
	cv::Mat bgModel, fgModel;
	cv::grabCut(image, result,
		rectangle,
		bgModel,
		fgModel,
		1,
		cv::GC_INIT_WITH_RECT
	);
	
	imshow("result", result);
	waitKey(0);
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	imshow("result-after", result);
	waitKey(0);
	// Generate output image
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	imshow("foreground", foreground);
	waitKey(0);
	image.copyTo(foreground, result); // bg pixels not copied
	imshow("foreground-merge", foreground);
	waitKey(0);
									  // draw rectangle on original image
	cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	// display result
	cv::namedWindow("Segmented Image");
	cv::imshow("Segmented Image", foreground);
	out = foreground.clone();

}

void CImageDector::runCanny(string strPath, Mat &out)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);
	Mat edge;
	blur(imageGray, edge, Size(3, 3));

	Canny(edge, edge, 50, 255, 3);
	imshow("result", edge);
	waitKey(0);
	out = edge.clone();
}

void CImageDector::runSobel(string strPath, Mat &out)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;
	//Compute X gradiant
	Sobel(imageGray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x/*alpha,beta*/); //改公式执行dst = abs(alpha*src + beta)
	imshow("x方向梯度", abs_grad_x);
	waitKey(0);
	//Compute Y
	Sobel(imageGray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("Y方向梯度", abs_grad_y);
	waitKey(0);
	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	imshow("整体Sobel", dst);
	waitKey(0);
	out = dst.clone();
}

void CImageDector::runLaplas(string strPath, Mat &out)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);
	Mat dst, abs_dst;
	Laplacian(imageGray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	imshow("Laplacian result", abs_dst);
	waitKey(0);
	out = abs_dst.clone();
}

void CImageDector::runSobel_Canny(string strPath, Mat &out,int low,int high)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);
	imshow("gray", imageGray);
	waitKey(0);
	/*
		计算SOBEL 梯度图，找出Canny 的上下阈值
	*/
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;
	//Compute X gradiant
	Sobel(imageGray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x/*alpha,beta*/); //改公式执行dst = abs(alpha*src + beta)
	imshow("x方向梯度", abs_grad_x);
	waitKey(0);
	//Compute Y
	Sobel(imageGray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("Y方向梯度", abs_grad_y);
	waitKey(0);
	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	imshow("整体Sobel", dst);

	Mat lowThread, highThread;
	cv::threshold(dst, lowThread, low, 255, cv::THRESH_BINARY_INV);
	cv::threshold(dst, highThread, high, 255, cv::THRESH_BINARY_INV);
	imshow("低阈值", lowThread);
	imshow("高阈值", highThread);
	waitKey(0);
	Canny(imageGray, dst, low, high);
	imshow("Canny result", dst);
	out = dst.clone();

}

void CImageDector::run_HoughLine(string strPath, Mat &out,int nMinCount)
{

	runSobel_Canny(strPath, out, 120, 200);
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);

	vector<cv::Vec2f>lines;
	/*
	
	image:边缘检测的输出图像. 它应该是个灰度图 (但事实上是个二值化图)

	lines:储存着检测到的直线的参数对  的容器 

	rho:参数极径  以像素值为单位的分辨率. 我们使用 1 像素.

	theta:参数极角  以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)

	threshold:检测 一条直线所需最少的的曲线交点

	srn and stn: 参数默认为0.
	*/
	int rho = 1;
	int theta = CV_PI / 180;
	int threshold = 100;

	HoughLines(out, lines, rho, theta, threshold,0,0);
	
	std::cout << "Lines detected: " << lines.size() << std::endl;

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(image, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
	}

	// Display the detected line image
	cv::namedWindow("Detected Lines with Hough");
	cv::imshow("Detected Lines with Hough", image);
	waitKey(0);

	
}


void CImageDector::run_HoughLineP(string strPath, Mat &out, int nMinCount)
{

	runSobel_Canny(strPath, out, 120, 200);
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
	imshow("cur", image);
	waitKey(0);

	vector<cv::Vec2f>lines;
	/*
	
	rho : 　参数极径  以像素值为单位的分辨率. 我们使用 1 像素.

	theta: 参数极角  以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)

	threshold: 检测 一条直线所需最少的的曲线交点 

	minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.线段的最小长度

	maxLineGap:线段上最近两点之间的阈值

	*/
	int rho = 1;
	int theta = CV_PI / 180;
	int threshold = 100;
	int minLinLength = 50;
	int maxLineGap = 10;
	HoughLinesP(out, lines, rho, theta, threshold, minLinLength, maxLineGap);

	std::cout << "Lines detected: " << lines.size() << std::endl;

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(image, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
	}

	// Display the detected line image
	cv::namedWindow("Detected Lines with Hough");
	cv::imshow("Detected Lines with Hough", image);
	waitKey(0);


}

void CImageDector::run_HoughCircle(string strPath, Mat &out, int dp,int para1,int para2)
{
	runSobel_Canny(strPath, out, 120, 200);
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();
		
	imshow("cur", image);
	waitKey(0);
	vector<Vec3f> circles;
	//int minRadis
	/*
	
	第一阶段：检测圆心

	1.1、对输入图像边缘检测；

	1.2、计算图形的梯度，并确定圆周线，其中圆周的梯度就是它的法线；

	1.3、在二维霍夫空间内，绘出所有图形的梯度直线，某坐标点上累加和的值越大，说明在该点上直线相交的次数越多，也就是越有可能是圆心；

	1.4、在霍夫空间的4邻域内进行非最大值抑制；

	1.5、设定一个阈值，霍夫空间内累加和大于该阈值的点就对应于圆心。

	第二阶段：检测圆半径

	2.1、计算某一个圆心到所有圆周线的距离，这些距离中就有该圆心所对应的圆的半径的值，这些半径值当然是相等的，并且这些圆半径的数量要远远大于其他距离值相等的数量；

	2.2、设定两个阈值，定义为最大半径和最小半径，保留距离在这两个半径之间的值，这意味着我们检测的圆不能太大，也不能太小；

	2.3、对保留下来的距离进行排序；

	2.4、找到距离相同的那些值，并计算相同值的数量；

	2.5、设定一个阈值，只有相同值的数量大于该阈值，才认为该值是该圆心对应的圆半径；

	2.6、对每一个圆心，完成上面的2.1～2.5步骤，得到所有的圆半径。

	*/
	dp = 1;//代表霍夫空间与原空间尺寸一致
	int minDist = 10;//minDist为圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
	int Para1 = 100; //param1为边缘检测时使用Canny算子的高阈值
	int Para2 = 30;//param2为步骤1.5和步骤2.5中所共有的阈值
	int nMinRadis = 10, nMaxRadis = 50;// minRadius和maxRadius为所检测到的圆半径的最小值和最大值
	HoughCircles(out, circles, CV_HOUGH_GRADIENT, dp, minDist, Para1, Para2, nMinRadis,  nMaxRadis);

	for (size_t i = 0; i < circles.size(); i++)
	{
		//提取出圆心坐标  
		Point center(round(circles[i][0]), round(circles[i][1]));
		//提取出圆半径  
		int radius = round(circles[i][2]);
		//圆心  
		circle(image, center, 3, Scalar(0, 255, 0), -1, 4, 0);
		//圆  
		circle(image, center, radius, Scalar(0, 0, 255), 3, 4, 0);
	}

	namedWindow("Circle", CV_WINDOW_AUTOSIZE);
	imshow("Circle", image);
	waitKey(0);
	out = image.clone();

}

void CImageDector::run_Harris(string strPath, Mat &out, int nThread)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();

	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image,imageGray);

	/*
	原理：
	【1】.局部窗口沿各方向移动，均产生明显变化的点
	【2】.图像局部曲率突变的点
	【3】.基本思想：从图像局部的小窗口观察图像特征，在各个方向移动都会导致图像灰度的明显变化，也就是说图像的梯度在各个方向有很大变化。

	步骤：
	1. 对于灰度图像 I，我们在窗口w(x,y)内，用x方向的偏移u和y方向的偏移v，扫过所有像素，得到一个图像灰度值的变化和。 E(u,v) = Sigma(I(x+u,y+v) - I(x,y))^2; 
	2. 对E(u,v) 做泰勒基数展开，简化后得到 E(u,v) = [u,v]M [
														  u,
														  v
														   ]

	3. 对每一个窗口计算得到一个分数R，根据R的大小来判定窗口内是否存在harris特征角。分数R  = det(M) + K*trace(M), 其中 det(M) = r1*R2, trace(M) = r1+r2;(r1,r2是M的特征值）
	4. k是一个指定值，这是一个经验参数，需要实验确定它的合适大小，通常它的值在0.05和0.5之间)

	OpenCV 函数：
	C++: void cornerHarris(InputArray src, OutputArray dst, int blockSize, int ksize, double k, intborderType=BORDER_DEFAULT )
	参数说明：
	src – 单通道8位或者浮点图像。
	dst – 存储 Harris 角的结果图像，它的格式为：CV_32FC1，图像大小和源图像一致。
	blockSize – 就是扫描时候窗口的大小。
	ksize – Sobel() 算子使用的值。
	k – 上面介绍的计算R时候的k参数值。通常在0.05~0.5之间
	borderType –像素插值方法。
	*/
	cornerHarris(imageGray, out, 2, 3, 0.1, BORDER_DEFAULT);

	Mat threadHold;
	threshold(out, threadHold, 0.00001, 255, THRESH_BINARY);
	imshow("结果", threadHold);
	out = threadHold.clone();
}

void CImageDector::run_HarrisEx(string strPath, Mat &out, int nThread)
{
	m_IP.LoadImage(0, strPath);
	Mat image = m_IP.GetImage();

	imshow("cur", image);
	waitKey(0);
	Mat imageGray;
	m_IP.GetGaryImage(image, imageGray);

	Mat correrStrength;
	cv::cornerHarris(imageGray, correrStrength,
		2,		// neighborhood size
		3,     // aperture size
		0.1);           // Harris parameter

	imshow("cornerHarris pre result", correrStrength);
	waitKey(0);

	// internal threshold computation
	double minStrength, maxStrength; // not used
	cv::minMaxLoc(correrStrength, &minStrength, &maxStrength);

	// local maxima detection
	Mat localMax;
	cv::Mat dilated;  // temporary image
	cv::dilate(correrStrength, dilated, cv::Mat());
	imshow("Dilated Corner", dilated);
	waitKey(0);
	cv::compare(correrStrength, dilated, localMax, cv::CMP_EQ);
	imshow("Dilated Corner compare", dilated);
	waitKey(0);
	//
	cv::Mat cornerMap;
	float qualityLevel = 0.01;
	// thresholding the corner strength
	float threshold = qualityLevel*maxStrength;
	Mat cornerTh;
	cv::threshold(correrStrength, cornerTh, threshold, 255, cv::THRESH_BINARY);
	imshow("cornerMap before convert", cornerTh);
	waitKey(0);
	// convert to 8-bit image
	cornerTh.convertTo(cornerMap, CV_8U);
	imshow("cornerMap after convert", cornerMap);
	waitKey(0);
	// non-maxima suppression
	cv::bitwise_and(cornerMap, localMax, cornerMap);
	imshow("CornerMap after bitwise", cornerMap);
	//获取角点
	std::vector<cv::Point> pts;
	for (int y = 0; y < cornerMap.rows; y++) 
	{

		const uchar* rowPtr = cornerMap.ptr<uchar>(y);

		for (int x = 0; x < cornerMap.cols; x++) 
		{

			// if it is a feature point
			if (rowPtr[x]) 
			{

				pts.push_back(cv::Point(x, y));
			}
		}
	}
	//将角点打印在图片上
	std::vector<cv::Point>::const_iterator it = pts.begin();

	// for all corners
	while (it != pts.end()) {

		// draw a circle at each corner location
		cv::circle(image, *it, 2, cv::Scalar(0, 0, 255), 1);
		++it;
	}
	imshow("检测结果", image);
	out = image.clone();

}