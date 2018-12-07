#include "stdafx.h"
#include "StructUsage.h"
#include <iostream>
using namespace std;



namespace
{
	//! [mandelbrot-escape-time-algorithm]
	int mandelbrot(const complex<float> &z0, const int max)
	{
		complex<float> z = z0;
		for (int t = 0; t < max; t++)
		{
			if (z.real()*z.real() + z.imag()*z.imag() > 4.0f) return t;
			z = z*z + z0;
		}

		return max;
	}
	//! [mandelbrot-escape-time-algorithm]

	//! [mandelbrot-grayscale-value]
	int mandelbrotFormula(const complex<float> &z0, const int maxIter = 500) {
		int value = mandelbrot(z0, maxIter);
		if (maxIter - value == 0)
		{
			return 0;
		}

		return cvRound(sqrt(value / (float)maxIter) * 255);
	}
	//! [mandelbrot-grayscale-value]

	//! [mandelbrot-parallel]
	class ParallelMandelbrot : public ParallelLoopBody
	{
	public:
		ParallelMandelbrot(Mat &img, const float x1, const float y1, const float scaleX, const float scaleY)
			: m_img(img), m_x1(x1), m_y1(y1), m_scaleX(scaleX), m_scaleY(scaleY)
		{
		}

		virtual void operator ()(const Range& range) const 
		{
			for (int r = range.start; r < range.end; r++)
			{
				int i = r / m_img.cols;
				int j = r % m_img.cols;

				float x0 = j / m_scaleX + m_x1;
				float y0 = i / m_scaleY + m_y1;

				complex<float> z0(x0, y0);
				uchar value = (uchar)mandelbrotFormula(z0);
				m_img.ptr<uchar>(i)[j] = value;
			}
		}

		ParallelMandelbrot& operator=(const ParallelMandelbrot &) {
			return *this;
		};

	private:
		Mat &m_img;
		float m_x1;
		float m_y1;
		float m_scaleX;
		float m_scaleY;
	};
	//! [mandelbrot-parallel]

	//! [mandelbrot-sequential]
	void sequentialMandelbrot(Mat &img, const float x1, const float y1, const float scaleX, const float scaleY)
	{
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				float x0 = j / scaleX + x1;
				float y0 = i / scaleY + y1;

				complex<float> z0(x0, y0);
				uchar value = (uchar)mandelbrotFormula(z0);
				img.ptr<uchar>(i)[j] = value;
			}
		}
	}
	//! [mandelbrot-sequential]
}



CStructUsage::CStructUsage()
{
}


CStructUsage::~CStructUsage()
{
}

void CStructUsage::ScalarUsage()
{
	/*
		typedef struct Scalar
		{
			double val[4];
		}Scalar;
		Scalar��һ���ɳ���Ϊ4��������ΪԪ�ع��ɵĽṹ�壬Scalar�����Դ洢�ĸ�ֵ��û���ṩ��ֵĬ����0
		Usage:
		Mat M(7,7,CV_32FC2,Scalar(1,3));
		����һ��2ͨ������ÿ��ͨ����ֵ��Ϊ��1,3�������Ϊ32��7��7�е�ͼ�����CV_32F��ʾÿ��Ԫ�ص�ֵ������Ϊ32λ��������C2��ʾͨ����Ϊ2��Scalar��1,3����ʾ�Ծ���ÿ��Ԫ�ض���ֵΪ��1,3������һ��ͨ���е�ֵ����1���ڶ���ͨ���е�ֵ����3
	*/
	Mat m(7, 7, CV_32FC2, Scalar(1, 3));
	cout << m << endl;

	Mat m1(7, 7, CV_32FC4, Scalar(1, 3));  //4 ͨ������2��ͨ������ֵ�Ļ�Ĭ��Ϊ0��
	cout << m1 << endl;

	Mat m3(256, 256, CV_32FC3, Scalar(255, 0, 0));
	//cout << m3 << endl;
	imshow("m3", m3);
}

void CStructUsage::MatUsage()
{
	/*
	1.Mat ��ֵʱ��ֻ�Ƕ࿽����һ�������ָ�룬������û�и���
	2.���Ҫ�������ݣ�����Ҫ�� mat.copy()  �� mat.clone()
	3.mat.isContinus() �����ã��ж����ݵĴ洢�Ƿ��������ģ�����������ģ�������� p++ ����������
	4.��������ΪCV_32FC2 ʱ���������ֱ����������һ�£������������Ϊɶ��
	*/
	Mat m3(10, 10, CV_8UC3, Scalar(255, 0, 0));
	
	long nRows = m3.rows * m3.channels();
	long nCols = m3.cols;
	uchar *p = m3.data;
	if (m3.isContinuous())
	{
		nCols *= nRows;
		for (long i=0;i < nCols;i ++)
		{
			cout << (int)*p << ",";
			p++;
		}
	}
	cout << endl;


	for (int i = 0; i < m3.rows; i++)
	{
		uchar* data = m3.ptr<uchar>(i);
		for (int j = 0; j < m3.cols * m3.channels(); j++)
		{
			//*data = 255;
			cout << (int)(data[j]) << ","; // ����Ҫת����int ,�������ַ����������̨��Ϊ��
			//d//ata++;

		}
		cout << endl;
	}
	cout << m3 << endl;
	imshow("m4", m3);

	//��ʼ��һ��kernel �ķ���
	Mat kernel = (Mat_<char>(3, 3)
		<<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0
		);
	cout << kernel;
	waitKey(0);

}

void CStructUsage::RNG_Usage()
{
	RNG rng;
	// always produces 0
	double a = rng.uniform(0, 1);
	cout << a << endl;
	// produces double from [0, 1)
	double a1 = rng.uniform((double)0, (double)1);
	cout << a1 << endl;
	// produces float from [0, 1)
	double b = rng.uniform(0.f, 1.f);
	cout << b << endl;
	// produces double from [0, 1)
	double c = rng.uniform(0., 1.);
	cout << c << endl;
	// may cause compiler error because of ambiguity:
	// RNG::uniform(0, (int)0.999999)? or RNG::uniform((double)0, 0.99999)?
	//double d = rng.uniform(0, 0.999999);
	//cout << d << endl;

	double e = rng.gaussian(20);
	cout << e;
	double F = rng.next();
	cout << F;
	
}

void CStructUsage::MakeBorder_Usage(Mat &img)
{
	RNG rng;
	int nLeft, nTop, nRight, nBottom = 0;
	nTop = (int)(0.5*img.rows);
	nLeft = (int)(0.5*img.cols);
	nRight = (int)(0.5*img.cols);
	nBottom = (int)(0.5*img.rows);
	Mat dst = img.clone();

	Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	copyMakeBorder(img, dst, nTop, nBottom, nLeft, nRight, BORDER_CONSTANT, value);
	imshow("dst_constant", dst);
	copyMakeBorder(img, dst, nTop, nBottom, nLeft, nRight, BORDER_REFLECT, value);
	imshow("dst_REFLECT", dst);
	waitKey(0);


}

void CStructUsage::Parallelize()
{
	Mat mandelbrotImg(4800, 5400, CV_8U);
	float x1 = -2.1f, x2 = 0.6f;
	float y1 = -1.2f, y2 = 1.2f;
	float scaleX = mandelbrotImg.cols / (x2 - x1);
	float scaleY = mandelbrotImg.rows / (y2 - y1);
	//! [mandelbrot-transformation]

	double t1 = (double)getTickCount();

#ifdef CV_CXX11

	//! [mandelbrot-parallel-call-cxx11]
	parallel_for_(Range(0, mandelbrotImg.rows*mandelbrotImg.cols), [&](const Range& range) {
		for (int r = range.start; r < range.end; r++)
		{
			int i = r / mandelbrotImg.cols;
			int j = r % mandelbrotImg.cols;

			float x0 = j / scaleX + x1;
			float y0 = i / scaleY + y1;

			complex<float> z0(x0, y0);
			uchar value = (uchar)mandelbrotFormula(z0);
			mandelbrotImg.ptr<uchar>(i)[j] = value;
		}
	});
	//! [mandelbrot-parallel-call-cxx11]

#else

	//! [mandelbrot-parallel-call]
	ParallelMandelbrot parallelMandelbrot(mandelbrotImg, x1, y1, scaleX, scaleY);
	parallel_for_(Range(0, mandelbrotImg.rows*mandelbrotImg.cols), parallelMandelbrot);
	//! [mandelbrot-parallel-call]

#endif

	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	cout << "Parallel Mandelbrot: " << t1 << " s" << endl;

	Mat mandelbrotImgSequential(4800, 5400, CV_8U);
	double t2 = (double)getTickCount();
	sequentialMandelbrot(mandelbrotImgSequential, x1, y1, scaleX, scaleY);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	cout << "Sequential Mandelbrot: " << t2 << " s" << endl;
	cout << "Speed-up: " << t2 / t1 << " X" << endl;

	imwrite("Mandelbrot_parallel.png", mandelbrotImg);
	imwrite("Mandelbrot_sequential.png", mandelbrotImgSequential);

	return ;
}