#include <opencv2/opencv.hpp>
#include <iostream>  
#include <time.h>

using namespace cv;
using namespace std;

#define WIDTH  852
#define HEIGHT 480

int main(int, char**)
{
	VideoCapture cap2("60fps.mp4");
	if (!cap2.isOpened())
	{
		printf("동영상 파일을 열수 없습니다. \n");
	}

	cap2.set(CAP_PROP_FRAME_WIDTH, WIDTH);
	cap2.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
	
	Mat frame1;

	int width = WIDTH;
	int height = HEIGHT;
	int count = 0;

	for (;;)
	{

		clock_t st = clock();

		//웹캡으로부터 한 프레임을 읽어옴  
		cap2 >> frame1;
		count++;
		float sigma = 1.6;
		if (count > 120) {
			if (frame1.data == NULL) {
				cout << "end" << endl;
				break;
			}

			Mat gray, edge;
			cv::cvtColor(frame1, gray, COLOR_BGR2GRAY); 

			GaussianBlur(gray, edge, Size(3, 3), 0, 0); 

			clock_t ho1 = clock();

			cout << "gray & blur time: " << ho1 - st << endl;

			Mat contours;
			Canny(edge, contours, 10, 350);   

			clock_t ho2 = clock();

			cout << "canny time: " << ho2 - ho1 << endl;

			

			const int diagonal = sqrt(width * width + height * height);

			//허프 변환 구현
			int HoughSpace_width = 180;
			int HoughSpace_height = diagonal * 2;
			int HoughSpace_size = HoughSpace_height * HoughSpace_width;

			int centerX = width / 2.0;
			int centerY = height / 2.0;

			int *HoughSpace = (int*)malloc(sizeof(int)*HoughSpace_size);
			memset(HoughSpace, 0, sizeof(int)*HoughSpace_size);
	

	
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					for (int angle = 0; angle < 180; angle++)
					{
						int r = (x - centerX) * cos(angle * CV_PI / 180) + (y - centerY) * sin(angle * CV_PI / 180);
						r += diagonal;
						if (contours.at<uchar>(y, x) > 0) {
							HoughSpace[r * HoughSpace_width + angle] += 1;   
						}
					}
				}
			}
			clock_t ho3 = clock();

			cout << "hough time: " << ho3 - ho2 << endl;

			//voting이 끝난 rtheta평면을 xy평면으로 되돌리고 선을 긋는 과정
			for (int r = 0; r < HoughSpace_height; r++) {
				for (int angle = 0; angle < HoughSpace_width; angle++)
				{
					if (HoughSpace[r * HoughSpace_width + angle] >= 150)
					{
						int x1, x2, y1, y2;
						if (angle >= asin(height / diagonal)*CV_PI / 180 && angle <= 180 - asin(height / diagonal)*CV_PI / 180)
						{
							x1 = 0;
							x2 = width;
							y1 = ((r - diagonal) - (x1 - centerX)*cos(angle * CV_PI / 180)) / sin(angle * CV_PI / 180) + centerY;
							y2 = ((r - diagonal) - (x2 - centerX)*cos(angle * CV_PI / 180)) / sin(angle * CV_PI / 180) + centerY;
						}
						else
						{
							y1 = 0;
							y2 = height;
							x1 = ((r - diagonal) - (y1 - centerY)*sin(angle * CV_PI / 180)) / cos(angle * CV_PI / 180) + centerX;
							x2 = ((r - diagonal) - (y2 - centerY)*sin(angle * CV_PI / 180)) / cos(angle * CV_PI / 180) + centerX;
						}

						cv::Point start_point(x1, y1);
						cv::Point end_point(x2, y2);
						line(frame1, start_point, end_point, cv::Scalar(255, 255, 0), 1);
					
					}
				}
			}

			clock_t ho4 = clock();

			cout << "drawing time: " << ho4 - ho3 << endl;
			cout << "total time: " << ho4 - st << endl << endl << endl << endl << endl;

			//imshow("canny", contours);

			imshow("video2", frame1);
		}

		//30ms 정도 대기하도록 해야 동영상이 너무 빨리 재생되지 않음.
		if (count > 120) {
			if (waitKey(30) == 27)
				break; //ESC키 누르면 종료  
		}
	}
}