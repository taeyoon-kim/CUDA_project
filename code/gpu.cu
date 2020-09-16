
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

//60 : 852 480
//70 : 1920 1080
//70fps2 1104 828

#define WIDTH 852
#define HEIGHT 480
#define TILE_WIDTH 32

using namespace cv;
using namespace std;

void GaussianEdge(unsigned char *src, unsigned char *edge);
__global__ void GaussianMask(unsigned char *output, const unsigned char *src, int rows, int columns);


__global__ void rgb_2_grey(unsigned char *greyImage, uchar3 *color, int rows, int columns);
__global__ void HoughTransform(unsigned char* src, int* houghspace, int centerX, int centerY, int diagonal);
__global__ void HoughTransform2(unsigned char* src, int* houghspace, int centerX, int centerY, int diagonal);

__device__ int xy2rtheta(int i, int j, int diagonal, int centerX, int centerY, int angle);
void colorToGray(Mat &src, unsigned char *output);

__global__ void rgb_2_grey(unsigned char *greyImage, uchar3 *color, int rows, int columns);



unsigned char *pHostContourImage;
unsigned char *pDevContourImage;

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
	//clock_t st = clock();

	int width = WIDTH;
	int height = HEIGHT;


	int *dev_houghspace;

	int count = 0;

	for (;;)
	{


		//웹캡으로부터 한 프레임을 읽어옴  
		cap2 >> frame1;
		count++;

		float sigma = 1.6;
		if (count > 120) {
			if (frame1.data == NULL) {
				cout << "end" << endl;
				break;
			}

			Mat gray(height, width, CV_8UC1), edge(height, width, CV_8UC1);
			clock_t st = clock();

			unsigned char *uGray = gray.data;
			unsigned char *uEdge = edge.data;

			//cvtColor(frame1, gray, COLOR_BGR2GRAY); //그레이 스케일로
			colorToGray(frame1, uGray);

			//cout << frame1.type() << endl;
			//imshow("grayVideio", gray);

			cout << "Size: "<< gray.cols << " ,  " << gray.rows << endl;
			//GaussianBlur(gray, edge, Size(5, 5), 0, 0); //실제 구현에서는 입력 영상 자체가 0.5로 스무딩되었다고 가정함.
			GaussianEdge(uGray, uEdge);

			clock_t ho1 = clock();

			cout << "gray & blur time: " << ho1 - st << endl;

			Mat contours(height, width, CV_8UC1);

			Canny(edge, contours, 10, 350); 

			clock_t ho2 = clock();

			cout << "canny time: " << ho2 - ho1 << endl;

			/*윤재*/


			const int diagonal = sqrt(width * width + height * height);

			//허프 변환 구현
			int HoughSpace_width = 180;
			int HoughSpace_height = diagonal * 2;
			int HoughSpace_size = HoughSpace_height * HoughSpace_width;

			int centerX = width / 2.0;
			int centerY = height / 2.0;

			int *HoughSpace = (int*)malloc(sizeof(int)*HoughSpace_size);
			memset(HoughSpace, 0, sizeof(int)*HoughSpace_size);

			pHostContourImage = new unsigned char[WIDTH * HEIGHT];


			int color = contours.step * contours.rows;
		
			cudaError_t cudaStatus = cudaSetDevice(0);

			cudaStatus = cudaMalloc((void**)&pDevContourImage, WIDTH * HEIGHT * sizeof(unsigned char));
			cudaStatus = cudaMalloc((void **)&dev_houghspace, sizeof(int)*HoughSpace_size);

			cudaStatus = cudaMemcpy(pDevContourImage, contours.ptr(), color, cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(dev_houghspace, HoughSpace, sizeof(int) * HoughSpace_size, cudaMemcpyHostToDevice);

			/*dim3 dimGrid((width - 1) / (TILE_WIDTH / 2) + 1, (height - 1) / (TILE_WIDTH/2)+1, 45);
			dim3 dimBlock(TILE_WIDTH / 2, TILE_WIDTH / 2, 4);*/

			dim3 dimGrid((width - 1) / (TILE_WIDTH) + 1, (height - 1) / (TILE_WIDTH )+ 1);
			dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

			for (int i = 0; i < 1; i++) {
			HoughTransform2 << <dimGrid, dimBlock >> > (pDevContourImage, dev_houghspace, centerX, centerY, diagonal);
			cudaDeviceSynchronize();
			}

			clock_t ho3 = clock();

			cout << "hough time: " << ho3 - ho2 << endl;

			cudaStatus = cudaMemcpy(HoughSpace, dev_houghspace, sizeof(int)*HoughSpace_size, cudaMemcpyDeviceToHost);

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


			Mat outputMedia = frame1;
			int ch = waitKey(30);
			if (ch == '1') outputMedia = frame1;
			else if (ch == '2') outputMedia = gray;
			else if (ch == '3') outputMedia = edge;
			else if (ch == '4') outputMedia = contours;


			imshow("video", outputMedia);
			//imshow("video", frame1);
			//imshow("video2", contours);
		}

		//30ms 정도 대기하도록 해야 동영상이 너무 빨리 재생되지 않음.
		if (count > 120) {
			if (waitKey(30) == 27)
				break; //ESC키 누르면 종료  
		}
	}
	return 0;
}


__global__ void HoughTransform(unsigned char* src, int* houghspace, int centerX, int centerY, int diagonal)
{
	//int i = blockIdx.y * 32 + threadIdx.y;
	//int j = blockIdx.x * 32 + threadIdx.x;

	int i = blockIdx.y * 16 + threadIdx.y;
	int j = blockIdx.x * 16 + threadIdx.x;
	int k = blockIdx.z * 4 + threadIdx.z;
	if (i < HEIGHT && j < WIDTH) {
		if (src[i * WIDTH + j] > 0) {
		//for (int angle = 0; angle < 180; angle++)

		//{
		int temp = xy2rtheta(i, j, diagonal, centerX, centerY, k);
		//int r = (j - centerX) * cos(angle * CV_PI / 180) + (i - centerY) * sin(angle * CV_PI / 180);
		//r += diagonal;
		//atomicAdd(&houghspace[r * 180 + angle], 1);   //voting
		//if (src[i*WIDTH + j] > 0)
			atomicAdd(&houghspace[temp], 1);
		//180 = houghspace_width 인데 대체함, 시간나면 인수 넣기
	}
	}
}


__device__ int xy2rtheta(int i, int j, int diagonal, int centerX, int centerY, int angle)
{
	int r = (j - centerX) * cos(angle * CV_PI / 180) + (i - centerY) * sin(angle * CV_PI / 180);
	r += diagonal;
	return r * 180 + angle;
}

__global__ void HoughTransform2(unsigned char* src, int* houghspace, int centerX, int centerY, int diagonal)
{
	int i = blockIdx.y * 32 + threadIdx.y;
	int j = blockIdx.x * 32 + threadIdx.x;

	if (i < HEIGHT && j < WIDTH) {
		for (int angle = 0; angle < 180; angle++)
		{
			int temp = xy2rtheta(i, j, diagonal, centerX, centerY, angle);
			int r = (j - centerX) * cos(angle * CV_PI / 180) + (i - centerY) * sin(angle * CV_PI / 180);
			r += diagonal;
			if (src[i * WIDTH + j] > 0) {

				atomicAdd(&houghspace[r * 180 + angle], 1);   //voting
															  //180 = houghspace_width 인데 대체함, 시간나면 인수 넣기
			}
		}
	}
}


void colorToGray(Mat &src, unsigned char *output)
{
	int color = WIDTH * HEIGHT * 3;
	int gray = WIDTH * HEIGHT;

	unsigned char *dev_output;
	uchar3 *dev_src;

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_src, color);
	cudaMalloc((void**)&dev_output, gray);

	//devsrc로 src를 color* 사이즈만큼 메모리 복사 
	cudaMemcpy(dev_src, src.ptr(), color, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((WIDTH - 1) / TILE_WIDTH + 1, (HEIGHT - 1) / TILE_WIDTH + 1);

	rgb_2_grey << <dimGrid, dimBlock >> > (dev_output, dev_src, HEIGHT, WIDTH);
	cudaDeviceSynchronize();

	cudaMemcpy(output, dev_output, gray, cudaMemcpyDeviceToHost);

	cudaFree(dev_src);
	cudaFree(dev_output);

}

__global__ void rgb_2_grey(unsigned char *greyImage, uchar3 *color, int rows, int columns)
{
	int rgb_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int rgb_y = blockIdx.y * TILE_WIDTH + threadIdx.y;

	if ((rgb_x < columns) && (rgb_y < rows)) {
		int rgb_ab = rgb_y*columns + rgb_x;
		uchar3 rgbImg = color[rgb_ab];
		greyImage[rgb_ab] = unsigned char((float(rgbImg.x))*0.299f + (float(rgbImg.y))*0.587f + (float(rgbImg.z))*0.114f);
	}
}

void GaussianEdge(unsigned char *src, unsigned char *edge)	//gray, edge
{

	int size = WIDTH * HEIGHT;

	unsigned char *dev_blur;
	unsigned char *dev_src;

	cudaMalloc((void**)&dev_src, size);
	cudaMalloc((void**)&dev_blur, size);

	cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((WIDTH - 1) / TILE_WIDTH + 1, (HEIGHT - 1) / TILE_WIDTH + 1);

	GaussianMask << <dimGrid, dimBlock >> > (dev_blur, dev_src, HEIGHT, WIDTH);
	cudaDeviceSynchronize();

	cudaMemcpy(edge, dev_blur, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_src);
	cudaFree(dev_blur);
}

__global__ void GaussianMask(unsigned char *output, const unsigned char *src, int rows, int columns)	//??, gray
{
	float gaussianMask[] = { 0.0000, 0.0000, 0.0002, 0.0000, 0.0000,
		0.0000, 0.0113, 0.0837, 0.0113, 0.0000,
		0.0002, 0.0837, 0.6187, 0.0837, 0.0002,
		0.0000, 0.0113, 0.0837, 0.0113, 0.0000,
		0.0000, 0.0000, 0.0002, 0.0000, 0.0000 };

	/*float gaussianMask[] = { 0.0304, 0.0501, 0, 0, 0 ,
	 0.0501, 0.1771, 0.0519, 0, 0 ,
	 0, 0.0519, 0.1771, 0.0519, 0 ,
	 0, 0, 0.0519, 0.1771, 0.0501 ,
	 0, 0, 0, 0.0501, 0.0304 };
*/

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

	if ((x > 2 || x < columns - 2) && (y > 2 || y < rows - 2))
	{
		float mask = 0;

		for (int r = 0; r < 5; r++)
		{
			for (int c = 0; c < 5; c++)
			{
				int idx = (y + r - 2) * WIDTH + x + c - 2;
				mask += gaussianMask[r * 5 + c] * src[idx];
			}
		}
			output[y*WIDTH + x] = (unsigned char)(mask * 0.9);

	}
	else
		output[y*WIDTH + x] = 0;

}