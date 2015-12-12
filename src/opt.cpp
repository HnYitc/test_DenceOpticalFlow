
#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include "opencv2/cudaoptflow.hpp"

using namespace cv;
using namespace cv::superres;
using namespace std;

int main()
{
        VideoCapture capture(0);
	Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_Farneback();

        double re_size = 0.3;
	Mat temp;

	Mat prev;
	capture >> temp;
        resize(temp,prev,Size(),re_size,re_size,INTER_LINEAR);

	while (waitKey(1) == -1)
	{

		Mat curr;
		capture >> temp;

        	resize(temp,curr,Size(),re_size,re_size,INTER_LINEAR);

		Mat flowX, flowY;
		//opticalFlow->calc(prev, curr, flowX, flowY);

		cuda::GpuMat gpuCurr(curr), gpuPrev(prev), gpuFlowX, gpuFlowY,gpuStatus;
		opticalFlow->calc(gpuPrev, gpuCurr, gpuFlowX, gpuFlowY);
		gpuFlowX.download(flowX);
		gpuFlowY.download(flowY);

		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);

		Mat hsvPlanes[3];		
		hsvPlanes[0] = angle;
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);

		hsvPlanes[1] = magnitude;
		hsvPlanes[2] = Mat::ones(magnitude.size(), CV_32F);

		Mat hsv;
		merge(hsvPlanes, 3, hsv);

		Mat flowBgr;
		cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

		cv::imshow("input", curr);
		cv::imshow("optical flow", flowBgr);

		prev = curr;
	}
        return 0;
}

