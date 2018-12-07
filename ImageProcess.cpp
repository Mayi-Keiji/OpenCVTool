#include "stdafx.h"
#include "ImageProcess.h"
#include <vector>
#include <ostream>

CImageProcess::CImageProcess()
{
}


CImageProcess::~CImageProcess()
{
}

bool CImageProcess::LoadImage(int nMode,string path)
{

	m_Img = imread(path);
	if (m_Img.empty())
	{
		return false;
	}
	else
		return true;

}
bool CImageProcess::SaveImage(int nMode, string path)
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //
	compression_params.push_back(100); //

	return imwrite(path, m_Img, compression_params);
}
Mat& CImageProcess::GetImage()
{
	return m_Img;
}

bool CImageProcess::GetGaryImage(Mat &img, Mat &gray)
{
	Mat gray2;
	//m_Img.convertTo(gray2, CV_32FC3);
	cout << img.channels() << endl;
	if (img.channels() <= 1)
	{
		return false;
	}
	cvtColor(img, gray, CV_RGB2GRAY);
	return true;
}

bool CImageProcess::TraverseImage(Mat &img)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k =0; k < img.channels(); k++)
			{
				cout << (int)img.at<Vec3b>(i, j)[k] << ", "; // 这里要转换成int ,否则以字符输出，控制台上为空
			}
			
		}
		cout << endl;
	}
	return true;
}

bool CImageProcess::TraverseImage1(Mat &img)
{
	for (int i = 0; i < img.rows; i++)
	{
		uchar* data = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols * img.channels(); j++)
		{
			*data = 255;
			cout << (int)*data << ","; // 这里要转换成int ,否则以字符输出，控制台上为空
			*data++;

		}
		cout << endl;
	}
	return true;
}

bool CImageProcess::ThreadHold(int nThread, Mat &gray, Mat &res, int nType)
{
	//int nThreadUse = 0;
	/*if (!(nType & CV_THRESH_OTSU))
	{
		nThreadUse = nThread;
	}*/
	threshold(gray, res, nThread, 255, nType);
	return true;
}

bool CImageProcess::Enrode(Mat &img,Mat &out,int nType, int nSize)
{

	Mat element = getStructuringElement(nType, Size(nSize, nSize));
	erode(img, out, element);
	return true;
}
bool CImageProcess::Dilate(Mat &img, Mat &out, int nType, int nSize)
{
	Mat element = getStructuringElement(nType, Size(nSize, nSize));
	dilate(img, out, element);
	return true;
}

bool CImageProcess::Close(Mat &img, Mat &out, int nType, int nSize)
{

	
	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_CLOSE, element);

	return true;
}

bool CImageProcess::Open(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_OPEN, element);

	return true;
}

bool CImageProcess::Gradient(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_GRADIENT, element);

	return true;
}

bool CImageProcess::TopHat(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_TOPHAT, element);

	return true;
}

bool CImageProcess::BlackHat(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_BLACKHAT, element);

	return true;
}
//找一个集合组合中固定形状的原点（中心位置）。 A-B ^ ~A-(W-B)
bool CImageProcess::HitOrMiss(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_BLACKHAT, element);

	return true;
}

bool CImageProcess::CreateKernel(int nSize, int ar[])
{
	//初始化一个kernel 的方法
	Mat kernel = (Mat_<char>(3, 3)
		<<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0
		);
	cout << kernel;

	return true;
}

bool CImageProcess::Blur(Mat &img, Mat &out, Filter_Type eType, int nSize, float fSigmaX, float fSigmaY)
{
	if (img.empty() || nSize == 0)
	{
		return false;
	}
	switch (eType)
	{
	case TYPE_MEAN:
		blur(img, out, Size(nSize, nSize));
		break;
	case TYPE_GUSSIAN:
		//fSigmaX: 高斯函数X方向的标准差
		//fSigmaY: 高斯函数Y方向的标准差
		GaussianBlur(img, out, Size(nSize, nSize), fSigmaX, fSigmaY);
		break;
	case TYPE_MEDIAN:
		medianBlur(img, out, nSize);
		break;
	case TYPE_bILATERAL:
	{
		float fSigmaColor = fSigmaX;
		float fSigmaSpace = fSigmaY;
		//nSize : 扫描过程中每个邻域的半径大小
		//fSigmaColor:: 颜色空间过滤器的sigma值，这个参数的值月大，表明该像素邻域内有月宽广的颜色会被混合到一起，产生较大的半相等颜色区域
		//fSigmaSpace: 坐标空间中滤波器的sigma值，如果该值较大，则意味着颜色相近的较远的像素将相互影响，从而使更大的区域中足够相似的颜色获取相同的颜色
		bilateralFilter(img, out, nSize, fSigmaColor, fSigmaSpace);
	}
		break;
	default:
		break;
	}
	return true;
}

bool CImageProcess::BlurH(Mat &img, Mat&out, int nSize)
{
	if (img.empty() || nSize == 0)
	{
		return false;
	}
	blur(img, out, Size(nSize, 1));
	return true;

}
bool CImageProcess::BlurV(Mat &img, Mat&out, int nSize)
{
	if (img.empty() || nSize == 0)
	{
		return false;
	}
	blur(img, out, Size(1, nSize));
	return true;
}

bool CImageProcess::Filter(Mat &img, Mat&out, Mat& kernel)
{
	filter2D(img, out, img.depth(), kernel);
	return true;

}

bool CImageProcess::ContrastBrightness(Mat &img, Mat &out, float alpha, float beta)
{
	if (img.empty())
	{
		return false;
	}
	out = img.clone();
	for (int i=0;i < img.rows; i++)
	{
		uchar* data = out.ptr<uchar>(i);
		for (int j = 0;j < img.cols*img.channels(); j++)
		{
			data[j] = saturate_cast<uchar>(alpha*data[j] + beta);
		}
	}
	
	return true;
}

bool CImageProcess::AddTwoImage(Mat &img1, Mat&img2, Mat&out, float fAlpha,int x,int y)
{
	float fBeta = 1 - fAlpha; 
	if (x == 0 && y == 0)
	{
		addWeighted(img1, fAlpha, img2, fBeta, 0, out);
	}
	else
	{
		Mat imgRoi(img1,Rect(x, y, img2.cols, img2.rows)); // 从一副图像获取ROI的方法
		//imgRoi=img1(Range(250,250+img1.rows),Range(200,200+img1.cols));
		addWeighted(imgRoi, fAlpha, img2, fBeta, 0, imgRoi);// 由于是浅拷贝，所以叠加之后结果还是在img1 上。
		out = img1.clone();

	}
	return true;
}

bool CImageProcess::Fourier(Mat &img, Mat &out)
{
	Mat padded;
	Mat gray;
	img = GetGaryImage(img, gray);
	int m = getOptimalDFTSize(gray.rows);
	int n = getOptimalDFTSize(gray.cols);
	copyMakeBorder(gray, padded, m - gray.rows, 0, n - gray.cols, 0, BORDER_CONSTANT, Scalar::all(0));

	//Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	//原图+一个空图合并为一个2通道图用于傅里叶变换
	merge(planes, 2, complexI);
	//做傅里叶变换
	dft(complexI, complexI);

	split(complexI, planes);
	//planes[0] 保存的实部， planes[1] 保存的虚部，梯度结果保存在planes[0]中
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	//magI += Scalar::all(1);

	log(magI, magI);

	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	//这个目的是将四个定点放到中心（即：1象限和3象限对调，2象限和4象限对调）
	Mat q0(magI, Rect(0, 0, cx, cy));
	Mat q1(magI, Rect(cx, 0, cx, cy));
	Mat q2(magI, Rect(0, cy, cx, cy));
	Mat q3(magI, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//对幅度谱归一化
	normalize(magI, magI, 0, 1, NORM_MINMAX);
	imshow("Input image", gray);
	imshow("Spectrum magnitude", magI);
	waitKey(0);
	return true;
}