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
				cout << (int)img.at<Vec3b>(i, j)[k] << ", "; // ����Ҫת����int ,�������ַ����������̨��Ϊ��
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
			cout << (int)*data << ","; // ����Ҫת����int ,�������ַ����������̨��Ϊ��
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
//��һ����������й̶���״��ԭ�㣨����λ�ã��� A-B ^ ~A-(W-B)
bool CImageProcess::HitOrMiss(Mat &img, Mat &out, int nType, int nSize)
{


	Mat element = getStructuringElement(nType,
		Size(nSize, nSize), Point(nSize / 2, nSize / 2));
	morphologyEx(img, out, MORPH_BLACKHAT, element);

	return true;
}

bool CImageProcess::CreateKernel(int nSize, int ar[])
{
	//��ʼ��һ��kernel �ķ���
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
		//fSigmaX: ��˹����X����ı�׼��
		//fSigmaY: ��˹����Y����ı�׼��
		GaussianBlur(img, out, Size(nSize, nSize), fSigmaX, fSigmaY);
		break;
	case TYPE_MEDIAN:
		medianBlur(img, out, nSize);
		break;
	case TYPE_bILATERAL:
	{
		float fSigmaColor = fSigmaX;
		float fSigmaSpace = fSigmaY;
		//nSize : ɨ�������ÿ������İ뾶��С
		//fSigmaColor:: ��ɫ�ռ��������sigmaֵ�����������ֵ�´󣬱������������������¿�����ɫ�ᱻ��ϵ�һ�𣬲����ϴ�İ������ɫ����
		//fSigmaSpace: ����ռ����˲�����sigmaֵ�������ֵ�ϴ�����ζ����ɫ����Ľ�Զ�����ؽ��໥Ӱ�죬�Ӷ�ʹ������������㹻���Ƶ���ɫ��ȡ��ͬ����ɫ
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
		Mat imgRoi(img1,Rect(x, y, img2.cols, img2.rows)); // ��һ��ͼ���ȡROI�ķ���
		//imgRoi=img1(Range(250,250+img1.rows),Range(200,200+img1.cols));
		addWeighted(imgRoi, fAlpha, img2, fBeta, 0, imgRoi);// ������ǳ���������Ե���֮����������img1 �ϡ�
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
	//ԭͼ+һ����ͼ�ϲ�Ϊһ��2ͨ��ͼ���ڸ���Ҷ�任
	merge(planes, 2, complexI);
	//������Ҷ�任
	dft(complexI, complexI);

	split(complexI, planes);
	//planes[0] �����ʵ���� planes[1] ������鲿���ݶȽ��������planes[0]��
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	//magI += Scalar::all(1);

	log(magI, magI);

	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	//���Ŀ���ǽ��ĸ�����ŵ����ģ�����1���޺�3���޶Ե���2���޺�4���޶Ե���
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
	//�Է����׹�һ��
	normalize(magI, magI, 0, 1, NORM_MINMAX);
	imshow("Input image", gray);
	imshow("Spectrum magnitude", magI);
	waitKey(0);
	return true;
}