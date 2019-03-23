#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace face;  

#define EIGEN_FACE 0
#define LBPH_FACE 1
#define RECOGNIZED_MODE 0
 
//GARY : face cut variable >>>
#define FACE_SIZE_MIN 20
#define SCALE_MODE 0
#define FACE_SAMPLE_W 80
#define FACE_SAMPLE_H 100
// Preprocess left & right sides of the face separately, in case there is stronger light on one side.
const bool preprocessLeftAndRightSeparately = true;   

Mat getPreprocessedFace(Mat srcImg, int desiredFaceWidth, int desiredFaceHeight, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye);
//GARY : face cut variable <<<

VideoCapture capture;

char *faceCascadeFilename = (char *)"./cascade/haarcascade_frontalface_alt.xml";
char *eyeCascadeFilename1 = (char *)"./cascade/haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
char *eyeCascadeFilename2 = (char *)"./cascade/haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.

CascadeClassifier cascade;
CascadeClassifier eyeCascade1;
CascadeClassifier eyeCascade2;
vector<Mat> images;
vector<int> labels;

int EIGEN_FACE_WIDTH = 280;
int EIGEN_FACE_HEIGHT = 280;
//ROI region.
int f_ROI_x = 200;
int f_ROI_y = 110;
int f_ROI_w = 280;
int f_ROI_h = 280;

string getLabelName(int label)
{
    string name;
    if(label == 0) name = "";
    else if(label == 1) name = "Gary";
    else if(label == 2) name = "Jerry";
    else if(label == 3) name = "Allen";
    else name = "";
    return name;
}

vector<Rect> detectAndDraw(Mat img)
{
    vector<Rect> faces;
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    equalizeHist( gray, gray );

    cascade.detectMultiScale(gray, faces);
    return faces;
}

int faceSamplePush(int sampleNum, const char* folderName, char* sampleName, int sampleLabel)
{
    for(int i=0; i<sampleNum; i++){
	char tmp[50];
	sprintf(tmp, "%simage%2d.jpg", folderName, i);
	    
	Mat mat = imread(tmp,1);                                                   
	if(mat.empty()) {
	    cout << "Couldn't read "<<tmp<< endl;
	    return -1;
	}
	
        Rect faceRect;
        Rect searchedLeftEye, searchedRightEye;
        Point leftEye, rightEye;
	Mat result = getPreprocessedFace(mat, FACE_SAMPLE_W, FACE_SAMPLE_H, preprocessLeftAndRightSeparately, 
			&faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

	if(result.cols > 0){
	    printf("%s OK\n", tmp);
	    images.push_back(result);                          
	    labels.push_back(sampleLabel);
	    //imshow(sampleName, result);    
	    //waitKey(1000);
	}
    }
    return 0;
}

int ProcessSampleRec(const char* saveLocation, int recNum)
{
    for(int i=0; i<recNum; i++){
	Mat frame;
	
	capture >> frame;
        if(frame.empty()) {
	    i--;	
	    continue;
        }

	Rect faceRect;
        Rect searchedLeftEye, searchedRightEye;
        Point leftEye, rightEye;
	Mat result = getPreprocessedFace(frame, FACE_SAMPLE_W, FACE_SAMPLE_H, preprocessLeftAndRightSeparately, 
			&faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

	if(result.cols > 0){
	    imshow("sample_windows", result);
	    imshow(saveLocation, frame);
	    int c = waitKey(10);
            if(c == 27 || c == 'q' || c == 'Q')
                break;
	    else if(c == 'c'){
		char tmp[50];
	        sprintf(tmp, "%simage%2d.jpg", saveLocation, i);
	        imwrite(tmp, frame);
	    }else
		i--;
	    
	    printf("get %d\n", i);
	}else{
	    i--;	
	    continue;
        }
    }
    return 0;
}

int main( int argc, const char** argv )
{
    int videoIdx = 0;

    if(!capture.open(videoIdx)){
            cout << "Capture from camera #" <<  videoIdx << " didn't work" << endl;
	    return -1;
    }
    
    if(!cascade.load(faceCascadeFilename)||!eyeCascade1.load(eyeCascadeFilename1)||!eyeCascade2.load(eyeCascadeFilename2))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    Ptr<FaceRecognizer> model;
    if(RECOGNIZED_MODE == EIGEN_FACE) model = createEigenFaceRecognizer(169, 4000);
    else if(RECOGNIZED_MODE == LBPH_FACE) model = createLBPHFaceRecognizer();createFisherFaceRecognizer();
    //Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(2, 16);
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

    //argv[1] : 1(capture sample and generate xml.)
    //argv[2] : the location for save capture sample.
    //argv[3] : capture sample number.
    if(argc>=2){
	if(argc>=4){
	    const char* location = argv[2];
	    int num              = atoi(argv[3]);
	    if(strcmp(argv[1], "1")==0) ProcessSampleRec(location, num);
	}
	else if(strcmp(argv[1], "2")==0){
	    //test if(faceSamplePush(2,  (char*)"./non/",       (char*)"",      0) < 0)return -1;	
	    if(faceSamplePush(80, (char*)"./gary_pic/",  (char*)"Gary",  1) < 0)return -1;	
	    if(faceSamplePush(80, (char*)"./jerry_pic/", (char*)"Jerry", 2) < 0)return -1;
	    if(faceSamplePush(9, (char*)"./allen_pic/", (char*)"Allen", 3) < 0)return -1;

	    printf("* start training...\n");
	    model->train(images, labels);
	    printf("* start training end...\n");
	    
	    model->save("./Eigenface_Model.xml");
	}
	return 0;
    } 

    if(capture.isOpened())
    {
	model->load("./Eigenface_Model.xml");
	
	for(;;)
        {
	    Mat frame;
            vector<Rect> faces;
	    
	    capture >> frame;
            if(frame.empty()) break;

	    Rect faceRect, searchedLeftEye, searchedRightEye;
            Point leftEye, rightEye;
	    Mat face = getPreprocessedFace(frame, FACE_SAMPLE_W, FACE_SAMPLE_H, preprocessLeftAndRightSeparately, 
			&faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

	    if(face.cols>0){
		//draw for user put his face.
	        rectangle(frame, Point(faceRect.x, faceRect.y),                          
			    Point(faceRect.x+faceRect.width, faceRect.y+faceRect.height), 
			    Scalar(0, 255, 0), 3, 8);

	        int predicted_label = -1;
	        double predicted_confidence = 0.0;
		double similarity = 0;

	        model->predict(face, predicted_label, predicted_confidence);

		if(RECOGNIZED_MODE == EIGEN_FACE) {
		    similarity = (double)predicted_confidence/((double)(face.rows*face.cols));
		    similarity = 1.0 - min(max(similarity, 0.0), 1.0);
		    predicted_confidence = (double)(similarity*100);
		}

		printf("label : %d, confidence : %f%s.\n", predicted_label, predicted_confidence, "%");
		if(predicted_confidence<=100&&predicted_confidence>0){
			char tmp[50];
			string str = getLabelName(predicted_label);
			sprintf(tmp, "%s %d%s", str.c_str(), (int)predicted_confidence, "%");
		        putText(frame, tmp, Point(faceRect.x, faceRect.y-15), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));
		}	    
		imshow("face", face);
		imshow("result", frame);
	    }else{
		imshow("result", frame);
	    }

            int c = waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }

    return 0;
}


//GDB****************************************************************************************************************************************
const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;         // Controls how tall the face mask is.

void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, CV_BGR2GRAY);
    else if (img.channels() == 4) cvtColor(img, gray, CV_BGRA2GRAY);
    else gray = img;

    // Possibly shrink the image, to run much faster.
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Shrink the image while keeping the same aspect ratio.
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else inputImg = gray;

    // Standardize the brightness and contrast to improve dark images.
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // Detect objects in the small grayscale image.
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // Enlarge the results if the image was temporarily shrunk before detection.
    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // Make sure the object is completely within the image, in case it was on a border.
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0) objects[i].x = 0;
        if (objects[i].y < 0) objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols) objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows) objects[i].y = img.rows - objects[i].height;
    }
    // Return with the detected face rectangles stored in "objects".
}

void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    int flags = CASCADE_FIND_BIGGEST_OBJECT;                   // Only search for just 1 object (the biggest in the image).
    Size minFeatureSize = Size(FACE_SIZE_MIN, FACE_SIZE_MIN);
    
    float searchScaleFactor = 1.1f;                            // How detailed should the search be. Must be larger than 1.0.
    // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    int minNeighbors = 4;

    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) largestObject = (Rect)objects.at(0);// Return the only detected object.
    else largestObject = Rect(-1,-1,-1,-1);// Return an invalid rect.
}

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // Skip the borders of the face, since it is usually just hair and ears, that we don't care about.
    // For "2splits.xml": Finds both eyes in roughly 60% of detected faces, also detects closed eyes.
    const float EYE_SX = 0.12f;
    const float EYE_SY = 0.17f;
    const float EYE_SW = 0.37f;
    const float EYE_SH = 0.36f;

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner

    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
    Rect leftEyeRect, rightEyeRect;

    // Return the search windows to the caller, if desired.
    if (searchedLeftEye) *searchedLeftEye = Rect(leftX, topY, widthX, heightY);
    if (searchedRightEye) *searchedRightEye = Rect(rightX, topY, widthX, heightY);

    // Search the left region, then the right region using the 1st eye detector.
    detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
    detectLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);

    // If the eye was not detected, try a different cascade classifier.
    if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) detectLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);

    // If the eye was not detected, try a different cascade classifier.
    if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);

    if (leftEyeRect.width > 0) {   // Check if the eye was detected.
        leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
        leftEyeRect.y += topY;
        leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
    }
    else {
        leftEye = Point(-1, -1);    // Return an invalid point
    }

    if (rightEyeRect.width > 0) { // Check if the eye was detected.
        rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
        rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
        rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
    }
    else {
        rightEye = Point(-1, -1);    // Return an invalid point
    }
}

void equalizeLeftAndRightHalves(Mat &faceImg)
{
    // It is common that there is stronger light from one half of the face than the other. In that case,
    // if you simply did histogram equalization on the whole face then it would make one half dark and
    // one half bright. So we will do histogram equalization separately on each face half, so they will
    // both look similar on average. But this would cause a sharp edge in the middle of the face, because
    // the left half and right half would be suddenly different. So we also histogram equalize the whole
    // image, and in the middle part we blend the 3 images together for a smooth brightness transition.

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) First, equalize the whole face.
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize the left half and the right half of the face separately.
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Left 25%: just use the left face.
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Mid-left 25%: blend the left face & whole face.
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Mid-right 25%: blend the right face & whole face.
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%: just use the right face.
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// end x loop
    }//end y loop
}

Mat getPreprocessedFace(Mat srcImg, int desiredFaceWidth, int desiredFaceHeight, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // Mark the detected face region and eye search regions as invalid, in case they aren't detected.
    if (storeFaceRect) storeFaceRect->width = -1;
    if (storeLeftEye) storeLeftEye->x = -1;
    if (storeRightEye) storeRightEye->x= -1;
    if (searchedLeftEye) searchedLeftEye->width = -1;
    if (searchedRightEye) searchedRightEye->width = -1;

    // Find the largest face.
    Rect faceRect;
    detectLargestObject(srcImg, cascade, faceRect, 320);

    // Check if a face was detected.
    if (faceRect.width > 0) {
        // Give the face rect to the caller if desired.
        if (storeFaceRect) *storeFaceRect = faceRect;

        Mat faceImg = srcImg(faceRect);    // Get the detected face image.

        // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
        Mat gray;
        if (faceImg.channels() == 3) cvtColor(faceImg, gray, CV_BGR2GRAY);
        else if (faceImg.channels() == 4) cvtColor(faceImg, gray, CV_BGRA2GRAY);
        else gray = faceImg;// Access the input image directly, since it is already grayscale.

        // Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
        Point leftEye, rightEye;
        detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);

        // Give the eye results to the caller if desired.
        if (storeLeftEye) *storeLeftEye = leftEye;
        if (storeRightEye) *storeRightEye = rightEye;

        // Check if both eyes were detected.
        if (leftEye.x >= 0 && rightEye.x >= 0) {
            // Make the face image the same size as the training images.

            // Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
            // line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
            // and not too far left or right of the face, etc.

            // Get the center between the 2 eyes.
            Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
            // Get the angle between the 2 eyes.
            double dy = (rightEye.y - leftEye.y);
            double dx = (rightEye.x - leftEye.x);
            double len = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

            // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
            const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
            // Get the amount we need to scale the image to be the desired fixed size we want.
            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
            double scale = desiredLen / len;
            // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
            Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
            // Shift the center of the eyes to be the desired center between the eyes.
            rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;
            // Rotate and scale and translate the image to the desired angle & size & position!
            // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
            Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
            warpAffine(gray, warped, rot_mat, warped.size());
            //imshow("warped", warped);

            // Give the image a standard brightness and contrast, in case it was too dark or had low contrast.
            if (!doLeftAndRightSeparately) equalizeHist(warped, warped);// Do it on the whole face.
            else equalizeLeftAndRightHalves(warped);// Do it seperately for the left and right sides of the face.
            //imshow("equalized", warped);

            // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
            Mat filtered = Mat(warped.size(), CV_8U);
            bilateralFilter(warped, filtered, 0, 20.0, 2.0);
            //imshow("filtered", filtered);

            // Filter out the corners of the face, since we mainly just care about the middle parts.
            // Draw a filled ellipse in the middle of the face-sized image.
            Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
            Point faceCenter = Point( desiredFaceWidth/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            Size size = Size( cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
            //imshow("mask", mask);

            // Use the mask, to remove outside pixels.
            Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
            /*
            namedWindow("filtered");
            imshow("filtered", filtered);
            namedWindow("dstImg");
            imshow("dstImg", dstImg);
            namedWindow("mask");
            imshow("mask", mask);
            */
            // Apply the elliptical mask on the face.
            filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
            //imshow("dstImg", dstImg);

            return dstImg;
        }
        /*
        else {
            // Since no eyes were found, just do a generic image resize.
            resize(gray, tmpImg, Size(w,h));
        }
        */
    }
    return Mat();
}
//gdb <<<
