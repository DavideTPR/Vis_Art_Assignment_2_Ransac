//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <cstdlib>


struct ArgumentList {
	std::string image_name;		    //!< image file name
};


bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}
		++i;
	}

	return true;
}

template <class T>
T bilinearAt(const cv::Mat& image, float r, float c)
{
    float s = c - std::floor(c);
    float t = r - std::floor(r);

    int floor_r = (int) std::floor(r);
    int floor_c = (int) std::floor(c);

    int ceil_r = (int) std::ceil(r);
    int ceil_c = (int) std::ceil(c);

    T value = (1 - s) * (1 - t) * image.at<T>(floor_r, floor_c)
                + s * (1 - t) * image.at<T>(floor_r, ceil_c)
                + (1 - s) * t * image.at<T>(ceil_r, floor_c)
                + s * t * image.at<T>(ceil_r, ceil_c);

    return value;
}



//------------------------ FUNZIONI ASSEGNAMENTO 1 ------------------------//


int convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
	//se la grandezza non è corretta esco
	if((kernel.cols*kernel.rows)%2 == 0)
	{
		return -1;
	}

	//se il numero di canali o il tipo di immagine non sono corretti esco
	if(image.channels() != 1 || image.type() != CV_8UC1)
	{
		return 1;
	}

	out = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0));
	float* ou = (float *)out.data;
	float* ker = (float*) kernel.data;
	int startR = (kernel.rows-1)/2;
	int startC = (kernel.cols-1)/2;


	for(int r = startR; r < image.rows-startR; r++)
	{
		for(int c = startC;  c < image.cols-startC; c++)
		{
			for(int kr = 0; kr < kernel.rows; kr++)
			{
				for(int kc = 0; kc < kernel.cols; kc++)
				{
					//essendo immagine di uc devo normalizzarla per far in modo che sia nel range tra 0 e 1 per le immagini float
					ou[c+r*out.cols] += (float) image.data[((c-startC+kc)+(r-startR+kr)*image.cols)*image.elemSize()] * 
					ker[(kc+kr*kernel.cols)];
				}
			}
			//ou[c+r*out.cols] = (float) image.data[(c+r*image.cols)*image.elemSize()]/255;
		}
	}


	return 0;
}


int convFFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
	//se la grandezza non è corretta esco
	if((kernel.cols*kernel.rows)%2 == 0)
	{
		return -1;
	}

	//se il numero di canali o il tipo di immagine non sono corretti esco
	/*if(image.channels() != 1 || image.type() != CV_32FC1)
	{
		return 1;
	}*/

	out = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0));
	float* ou = (float *)out.data;
	float* ker = (float*) kernel.data;
	float* in = (float*) image.data;
	int startR = (kernel.rows-1)/2;
	int startC = (kernel.cols-1)/2;


	for(int r = startR; r < image.rows-startR; r++)
	{
		for(int c = startC;  c < image.cols-startC; c++)
		{
			for(int kr = 0; kr < kernel.rows; kr++)
			{
				for(int kc = 0; kc < kernel.cols; kc++)
				{
					//essendo immagine di uc devo normalizzarla per far in modo che sia nel range tra 0 e 1 per le immagini float
					ou[c+r*out.cols] += (float) in[((c-startC+kc)+(r-startR+kr)*image.cols)] * 
					ker[(kc+kr*kernel.cols)];
				}
			}
			//ou[c+r*out.cols] = (float) image.data[(c+r*image.cols)*image.elemSize()]/255;
		}
	}


	return 0;
}

//ESERCIZIO 1 BIS
int conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
	if((kernel.cols*kernel.rows)%2 == 0)
	{
		return -1;
	}

	if(image.channels() != 1 || image.type() != CV_8UC1)
	{
		return 1;
	}

	out = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
	//float* ou = (float *)out.data;
	float* ker = (float*) kernel.data;
	//matrice che conterrà tutti i valori che verranno modificati col 
	//constrast stretching, convertiti e messi nella matrice risultante
	float* tmp = (float *)malloc(image.rows * image.cols*sizeof(float));
	int startR = (kernel.rows-1)/2;
	int startC = (kernel.cols-1)/2;

	float sum;
	float min = 255;
	float max = 0;

	for(int r = startR; r < image.rows-startR; r++)
	{
		for(int c = startC;  c < image.cols-startC; c++)
		{
			sum = 0;
			for(int kr = 0; kr < kernel.rows; kr++)
			{
				for(int kc = 0; kc < kernel.cols; kc++)
				{
					sum += (float)image.data[((c-startC+kc)+(r-startR+kr)*image.cols)*image.elemSize()] * 
					ker[(kc+kr*kernel.cols)];
				}
			}

			//cerco massimo e minimo per poter applicare constrast stretching
			if(sum < min)
			{
				min = sum;
			}

			if(sum > max)
			{
				max = sum;
			}
			tmp[(c+r*out.cols)] = sum;
			
			//out.data[(c+r*out.cols) * out.elemSize()] = (unsigned char) sum;
		}
	}

	//std::cout << "MINIMO:  " << (float)min << std::endl;
	//std::cout << "MASSIMO:  " << (float)max << std::endl;

	//constrast stretching
	float val;
	for(int r = startR; r < image.rows-startR; r++)
	{
		for(int c = startC;  c < image.cols-startC; c++)
		{
			val =  ((255 * (tmp[(c+r*out.cols)] - min))/(max - min));

			if(val < 0)
				val = 0;
			else
				if (val > 255)
					val = 255;

			out.data[(c+r*out.cols) * out.elemSize()] = (unsigned char) val;
		}
	}

	free(tmp);

	return 0;
}


void trasponi(const cv::Mat& mat1, cv::Mat& mat2)
{
	mat2 = cv::Mat(mat1.cols, mat1.rows, mat1.type());

	float * in = (float*) mat1.data;
	float * out = (float*) mat2.data;

	for(int r = 0; r < mat2.rows; r++)
	{
		for(int c = 0; c < mat2.cols; c++)
		{
			out[c+r*mat2.cols] = in[r+c*mat2.rows];
		}
	}
}

int constStre(const cv::Mat& in, cv::Mat& out)
{
	out = cv::Mat(in.rows, in.cols, CV_8UC1, cv::Scalar(0));
	float* tmp = (float *)malloc(in.rows * in.cols*sizeof(float));

	float * input = (float*) in.data;

	float min = 255;
	float max = 0;

	for(int r = 0; r < in.rows; r++)
	{
		for(int c = 0;  c < in.cols; c++)
		{

			if(input[(c+r*out.cols)] < min)
			{
				min = input[(c+r*out.cols)];
			}

			if(input[(c+r*out.cols)] > max)
			{
				max = input[(c+r*out.cols)];
			}
			
		}
	}

	//constrast stretching
	float val;
	for(int r = 0; r < in.rows; r++)
	{
		for(int c = 0;  c < in.cols; c++)
		{
			val =  ((255 * (tmp[(c+r*out.cols)] - min))/(max - min));

			if(val < 0)
				val = 0;
			else
				if (val > 255)
					val = 255;

			out.data[(c+r*out.cols) * out.elemSize()] = (unsigned char) val;
		}
	}

}


int gaussianKernel(float sigma, int radius, cv::Mat& kernel)
{
	kernel = cv::Mat((2*radius)+1 , (2*radius)+1, CV_32FC1);
	float* ker = (float*) kernel.data;
	int startR = (kernel.rows-1)/2;
	int startC = (kernel.cols-1)/2;
	float sum = 0;

	for(int kr = 0; kr < kernel.rows; kr++)
	{
		for(int kc = 0; kc < kernel.cols; kc++)
		{
			ker[(kc+kr*kernel.cols)] = (float)(1/(2 * M_PI * pow(sigma, 2)))*exp(-((pow((kc-startC), 2) + pow((kr-startR), 2))/(2 * pow(sigma, 2))));
			sum += ker[(kc+kr*kernel.cols)];
		}
	}

	return 0;
}

//-------------------------------------------------------------------------//

void harrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{
/**********************************
 *
 * PLACE YOUR CODE HERE
 *
 *
 *
 * E' ovviamente viatato utilizzare un detector di OpenCv....
 *
 */

	float kerD[3] = {-1,0,1};

	cv::Mat kernelH = cv::Mat(1, 3, CV_32FC1, kerD);
	cv::Mat kernelV;

	cv::Mat kernelG;
	cv::Mat kernelGX;
	cv::Mat kernelGY;

	cv::Mat imageH;
	cv::Mat imageV;
	cv::Mat imageHH;
	cv::Mat imageVV;
	cv::Mat imageHV;

	cv::Mat imageH2G;
	cv::Mat imageV2G;
	cv::Mat imageHVG;

	cv::Mat teta;

	cv::Mat imgShow;

	trasponi(kernelH, kernelV);

	convFloat(image, kernelH, imageH);
	convFloat(image, kernelV, imageV);

	//cv::Sobel(image, imageH, CV_32F, 1, 0, 3, 1, 0);

	imageH.convertTo(imgShow, CV_8UC1);
	//cv::cvtColor(imageH, imgShow, CV_8U);
	//constStre(imageH, imgShow);

	cv::namedWindow("H", cv::WINDOW_NORMAL);
	cv::imshow("H", imgShow);

	imageV.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("V", cv::WINDOW_NORMAL);
	cv::imshow("V", imgShow);

	//--------------------------------------------- MOLTIPLICAZIONI
	imageHH = imageH.mul(imageH);

	imageHH.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("HH", cv::WINDOW_NORMAL);
	cv::imshow("HH", imgShow);

	imageVV = imageV.mul(imageV);

	imageVV.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("VV", cv::WINDOW_NORMAL);
	cv::imshow("VV", imgShow);


	imageHV = imageH.mul(imageV);

	imageHV.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("HV", cv::WINDOW_NORMAL);
	cv::imshow("HV", imgShow);

	//--------------------------------------------- GAUSS
	gaussianKernel(1,2,kernelG);

	convFFloat(imageHH, kernelG, imageH2G);
	convFFloat(imageVV, kernelG, imageV2G);
	convFFloat(imageHV, kernelG, imageHVG);

	imageH2G.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("HHG", cv::WINDOW_NORMAL);
	cv::imshow("HHG", imgShow);

	imageV2G.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("VVG", cv::WINDOW_NORMAL);
	cv::imshow("VVG", imgShow);

	imageHVG.convertTo(imgShow, CV_8UC1);

	cv::namedWindow("HVG", cv::WINDOW_NORMAL);
	cv::imshow("HVG", imgShow);

	//std::cout << kernelG << std::endl;

	teta = (imageH2G.mul(imageV2G)) - imageHVG.mul(imageHVG)- alpha*((imageH2G + imageV2G).mul((imageH2G + imageV2G)));

	cv::Mat adjMap;
	cv::Mat falseColorsMap;
	double minr,maxr;
	cv::minMaxLoc(teta, &minr, &maxr);
	cv::convertScaleAbs(teta, adjMap, 255 / (maxr-minr));
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
	cv::namedWindow("response1", cv::WINDOW_NORMAL);
	cv::imshow("response1", falseColorsMap);

	//std::cout << "-----------MIN:" << minr << std::endl << "-----------MAX:" << maxr << std::endl;

	for(int r = 1; r < teta.rows-1; r++)
	{
		for(int c = 1; c < teta.cols-1; c++)
		{
			if(teta.at<float>(r,c) > harrisTh)
			{
				if(teta.at<float>(r,c) > teta.at<float>(r-1,c-1) &&
				   teta.at<float>(r,c) > teta.at<float>(r-1,c) &&
				   teta.at<float>(r,c) > teta.at<float>(r-1,c+1) &&
				   teta.at<float>(r,c) > teta.at<float>(r,c-1) &&
				   teta.at<float>(r,c) > teta.at<float>(r,c+1) &&
				   teta.at<float>(r,c) > teta.at<float>(r+1,c-1) &&
				   teta.at<float>(r,c) > teta.at<float>(r+1,c) &&
				   teta.at<float>(r,c) > teta.at<float>(r+1,c+1))
				   {
					   keypoints0.push_back(cv::KeyPoint(float(c), float(r), 5));
				   }
			}
		}
	}
}

void findHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::Point2f> & inliers_best0, std::vector<cv::Point2f> & inliers_best1)
{
	/**********************************
	 *
	 * PLACE YOUR CODE HERE
	 *
	 *
	 *
	 *
	 * E' possibile utilizzare:
	 * 		cv::findHomography(sample1, sample0, 0)
	 *
	 * E' vietato utilizzare:
	 * 		cv::findHomography(sample1, sample0, CV_RANSAC)
	 *
	 */

	cv::Mat point = cv::Mat(3, 1, 6);
	cv::Mat res;
	std::vector<cv::Point2f> sample0;
	std::vector<cv::Point2f> sample1;
	std::vector<cv::Point2f> c0;
	std::vector<cv::Point2f> c1;
	int inliers = 0;
	int count;
	int tmp;
	int old;

	int oldV[4] = {-1};

	srand(time(NULL));

	for(int k=0; k<N; k++)
	{
		tmp = 0;
		old = -1;
		int oldV[4] = {-1};

		for(int i=0; i < sample_size; i++)
		{
			do
			{
				
				tmp = (rand() % points1.size());

			}while(tmp == oldV[0] || tmp == oldV[1] || tmp == oldV[2] || tmp == oldV[3]);

			//std::cout << "T M P  --" << tmp << std::endl;

			oldV[i] = tmp;
			sample0.push_back(points0[tmp]);
			sample1.push_back(points1[tmp]);
			//std::cout << points1[tmp] << "--" << points0[tmp] << "--" << tmp << std::endl;

		}

		H = cv::findHomography(cv::Mat(sample0), cv::Mat(sample1), 0);

		count = 0;

		double* homography = (double *)H.data;

		for(int i=0; i < points0.size(); i++)
		{
			cv::Point3f ps2;

			ps2.x = homography[0]*points0[i].x + homography[1]*points0[i].y +homography[2];
			ps2.y = homography[3]*points0[i].x + homography[4]*points0[i].y +homography[5];
			ps2.z = homography[6]*points0[i].x + homography[7]*points0[i].y +homography[8];

			//std::cout << points1[i].x << " -X- " << ps2.x/ps2.z << std::endl;
			//std::cout << points1[i].y << " -Y- " << ps2.y/ps2.z << std::endl;

			//std::cout << "--Dist:" << sqrt(pow(points1[i].x - ps2.x/ps2.z, 2) + pow(points1[i].y - ps2.y/ps2.z, 2)) << std::endl;

			if(sqrt(pow(points1[i].x - ps2.x/ps2.z, 2) + pow(points1[i].y - ps2.y/ps2.z, 2)) < epsilon)
			{
				count++;
				c0.push_back(points0[i]);
				c1.push_back(points1[i]);
				//std::cout << ps2.x/ps2.z << " -X- " << ps2.x/ps2.y << std::endl;
				//std::cout << sample1[i].x << " -X- " << ps2.x/ps2.z << std::endl;
				//std::cout << sample1[i].y << " -Y- " << ps2.y/ps2.z << std::endl;
			}
		}

		//std::cout << count << std::endl;

		if(count > inliers)
		{
			inliers_best0.clear();
			inliers_best1.clear();
			inliers = count;
			for(int i=0; i < c0.size(); i++)
			{
				inliers_best0.push_back(c0[i]);
				inliers_best1.push_back(c1[i]);
			}
		}

		sample0.clear();
		sample1.clear();
		c0.clear();
		c1.clear();
	}
	//std::cout << inliers << std::endl;
	//std::cout << inliers_best0 << std::endl;
	//std::cout << inliers_best1 << std::endl;
	H = cv::findHomography( cv::Mat(inliers_best1), cv::Mat(inliers_best0), 0);

}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	//vettore delle immagini di input
	std::vector<cv::Mat> imageRGB_v;
	//vettore delle immagini di input grey scale
	std::vector<cv::Mat> image_v;

	std::cout<<"Simple image stitching program."<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);

		cv::Mat im = cv::imread(frame_name);
		if(im.empty())
		{
			break;
		}

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		//save RGB image
		imageRGB_v.push_back(im);

		//save grey scale image for processing
		cv::Mat im_grey(im.rows, im.cols, CV_8UC1);
		cvtColor(im, im_grey, CV_RGB2GRAY);
		image_v.push_back(im_grey);

		frame_number++;
	}

	if(image_v.size()<2)
	{
		std::cout<<"At least 2 images are required. Exiting."<<std::endl;
		return 1;
	}

	int image_width = image_v[0].cols;
	int image_height = image_v[0].rows;

	////////////////////////////////////////////////////////
	/// HARRIS CORNER
	//
	float alpha = 0.04;
	float harrisTh = 45000000;    //da impostare in base alla propria implementazione

	std::vector<cv::KeyPoint> keypoints0, keypoints1;

	harrisCornerDetector(image_v[0], keypoints0, alpha, harrisTh);
	harrisCornerDetector(image_v[1], keypoints1, alpha, harrisTh);
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DESCRITTORI E MATCHES
	//
    int briThreshl=30;
    int briOctaves = 3;
    int briPatternScales = 1.0;
	cv::Mat descriptors0, descriptors1;

	//dichiariamo un estrattore di features di tipo BRISK
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
    //calcoliamo il descrittore di ogni keypoint
    extractor->compute(image_v[0], keypoints0, descriptors0);
    extractor->compute(image_v[1], keypoints1, descriptors1);

    //associamo i descrittori tra me due immagini
    std::vector<std::vector<cv::DMatch> > matches;
	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
	matcher.radiusMatch(descriptors0, descriptors1, matches, image_v[0].cols*0.2);

    std::vector<cv::Point2f> points[2];
    for(unsigned int i=0; i<matches.size(); ++i)
      {
        if(!matches.at(i).empty())
          {
                points[0].push_back(keypoints0.at(matches.at(i).at(0).queryIdx).pt);
                points[1].push_back(keypoints1.at(matches.at(i).at(0).trainIdx).pt);
          }
      }
	////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
    // CALCOLO OMOGRAFIA
    //
    //
    // E' obbligatorio implementare RANSAC.
    //
    // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
    //
    // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
    //
    //
    cv::Mat H;            //omografia finale
	std::vector<cv::Point2f> inliers_best[2]; //inliers
    if(points[1].size()>=4)
    {
    	int N=100;            //numero di iterazioni di RANSAC
    	float epsilon = 0.5;  //distanza per il calcolo degli inliers
    	int sample_size = 4;  //dimensione del sample

    	//
    	//
    	// Abilitate questa funzione una volta che quella di opencv funziona
    	//
    	//
    	findHomographyRansac(points[1], points[0], N, epsilon, sample_size, H, inliers_best[0], inliers_best[1]);
    	//
    	//
    	//
    	//
    	//

    	std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    	std::cout<<"Num inliers / match totali "<<inliers_best[0].size()<<" / "<<points[0].size()<<std::endl;

    	//
    	//
    	// Rimuovere questa chiamata solo dopo aver verificato che i vostri corner di Harris generano una omografia corretta
    	//
    	//
    	//H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), CV_RANSAC );
    	//
    	//
    	//
    	//
    	//
    }
    else
    {
    	std::cout<<"Non abbastanza matches per calcolare H!"<<std::endl;
    	H = (cv::Mat_<double>(3, 3 )<< 1.0, 0.0, 0.0,
    		                           0.0, 1.0, 0.0,
			                           0.0, 0.0, 1.0);
    }

    std::cout<<"H"<<std::endl<<H<<std::endl;
    cv::Mat H_inv = H.inv();///H.at<double>(2,2);
    std::cout<<"H_inverse "<<std::endl<<H_inv<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DELLA DIMENSIONE DELL'IMMAGINE FINALE
    //
    cv::Mat p = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    cv::Mat tl = H*p;
    tl/=tl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, image_height-1, 1);
    cv::Mat br = H*p;
    br/=br.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << 0, image_height-1, 1);
    cv::Mat bl = H*p;
    bl/=bl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, 0, 1);
    cv::Mat tr = H*p;
    tr/=tr.at<double>(2,0);

    int min_warped_r = std::min(std::min(tl.at<double>(1,0), bl.at<double>(1,0)),std::min(tr.at<double>(1,0), br.at<double>(1,0)));
    int min_warped_c = std::min(std::min(tl.at<double>(0,0), bl.at<double>(0,0)),std::min(tr.at<double>(0,0), br.at<double>(0,0)));

    int max_warped_r = std::max(std::max(tl.at<double>(1,0), bl.at<double>(1,0)),std::max(tr.at<double>(1,0), br.at<double>(1,0)));
    int max_warped_c = std::max(std::max(tl.at<double>(0,0), bl.at<double>(0,0)),std::max(tr.at<double>(0,0), br.at<double>(0,0)));

    int min_final_r = std::min(min_warped_r,0);
    int min_final_c = std::min(min_warped_c,0);

    int max_final_r = std::max(max_warped_r,image_height-1);
    int max_final_c = std::max(max_warped_c,image_width-1);

    int width_final = max_final_c-min_final_c+1;
    int height_final = max_final_r-min_final_r+1;

    std::cout<<"width_final "<<width_final<<" height_final "<<height_final<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO IMMAGINE FINALE
    //
    cv::Mat outwarp(height_final, width_final, CV_8UC3, cv::Scalar(0,0,0));

    //copio l'immagine 0 sul nuovo piano immagine, e' solo uno shift
    imageRGB_v[0].copyTo(outwarp(cv::Rect(std::max(0,-min_warped_c), std::max(0,-min_warped_r), image_width, image_height)));

    //copio l'immagine 1 nel piano finale
    //in questo caso uso la trasformazione prospettica
    for(int r=0;r<height_final;++r)
    {
        for(int c=0;c<width_final;++c)
        {
        	cv::Mat p = (cv::Mat_<double>(3, 1) << c+std::min(0,min_warped_c), r+std::min(0,min_warped_r), 1);
        	cv::Mat pi = H_inv*p;
        	pi/=pi.at<double>(2,0);

        	if(int(pi.at<double>(1,0))>=1 && int(pi.at<double>(1,0))<image_height-1 && int(pi.at<double>(0,0))>=1 && int(pi.at<double>(0,0))<image_width-1)
        	{
        		cv::Vec3b pick = bilinearAt<cv::Vec3b>(imageRGB_v[1], pi.at<double>(1,0), pi.at<double>(0,0));

        		//media
        		if(outwarp.at<cv::Vec3b>(r,c) != cv::Vec3b(0.0))
        			outwarp.at<cv::Vec3b>(r,c) =  (outwarp.at<cv::Vec3b>(r,c)*0.5 + pick*0.5);
        		else
        			outwarp.at<cv::Vec3b>(r,c) = pick;
        	}
        }
    }
	////////////////////////////////////////////////////////

	////////////////////////////
	//WINDOWS
	//
    for(unsigned int i = 0;i<keypoints0.size();++i)
    	cv::circle(imageRGB_v[0], cv::Point(keypoints0[i].pt.x , keypoints0[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );

    for(unsigned int i = 0;i<keypoints1.size();++i)
    	cv::circle(imageRGB_v[1], cv::Point(keypoints1[i].pt.x , keypoints1[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );

	cv::namedWindow("KeyPoints0", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints0", imageRGB_v[0]);

	cv::namedWindow("KeyPoints1", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints1", imageRGB_v[1]);

    cv::Mat matchsOutput(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutput(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutput(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<points[0].size(); ++i)
    {
    	cv::Point2f p2shift = points[1][i];
    	p2shift.x+=imageRGB_v[0].cols;
    	cv::circle(matchsOutput, points[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutput, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutput, points[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches", cv::WINDOW_NORMAL);
	cv::imshow("Matches", matchsOutput);

    cv::Mat matchsOutputIn(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutputIn(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutputIn(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<inliers_best[0].size(); ++i)
    {
    	cv::Point2f p2shift = inliers_best[1][i];
    	p2shift.x+=image_width;
    	cv::circle(matchsOutputIn, inliers_best[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutputIn, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutputIn, inliers_best[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches Inliers", cv::WINDOW_NORMAL);
	cv::imshow("Matches Inliers", matchsOutputIn);

	cv::namedWindow("Outwarp", cv::WINDOW_AUTOSIZE);
	cv::imshow("Outwarp", outwarp);

	cv::waitKey(0);
	////////////////////////////

	return 0;
}
