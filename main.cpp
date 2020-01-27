/*Sean Nishi 2019
Face + body detector that actually does something. For home security purposes.
*/

/*
WHAT IS THIS?
This software detects people and their faces and takes a picture of that person as a record.

FUTURE FEATURES
- cross reference detected person with a database of people
- have camera follow detected person. Who should it follow when multiple people are detected?
-> maybe one camera follow a new person and a second camera follow me or person they recognize?
-> can maybe leave this one out and place cameras in positions that overlap
- 

KNOWN ISSUES
- video is sometimes laggy because it has to read from the camera every loop
-> workaround is to fill a buffer with frames and read from those frames
https://answers.opencv.org/question/29957/highguivideocapture-buffer-introducing-lag/post-id-38217/
https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer
- sometimes face and profile face classifiers classify the same thing. Leads to two pictures being taken.



*/

//opencv libs
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

//std libs
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <time.h>

/********************************************************************************************************************************/
//function declarations
int recording(cv::Mat frame, cv::VideoCapture real_time);
int camera_disconnected(cv::Mat frame, cv::VideoCapture real_time);

/********************************************************************************************************************************/
//functions
int main(int argc, char* argv[]) {
	bool camera_on = false;
	cv::Mat frame;

	//windows which recording will be displayed
	cv::namedWindow("Face Detection", CV_WINDOW_KEEPRATIO);

	//input from camera
	cv::VideoCapture real_time(0);
	if (!real_time.isOpened()) {
		std::cout << "ERROR: no video device detected" << std::endl;
		camera_on = false;
	}
	else {
		camera_on = true;
	}

	//infinite loop, even if camera is disconnected, program wont crash, will wait for camera to be plugged in again
	//not making return value a bool because maybe want different return values in the future.
	while (true) {
		if (camera_on) {
			//make the vars pointers so we aren't passing by value, instead pass by reference?
			if (recording(frame, real_time) > 0)
				camera_on = false;
		}
		else {
			//video was interrupted, await reconnect
			if (camera_disconnected(frame, real_time) < 0)
				camera_on = true;
		}
		//exit program
		if (cv::waitKey(10) == 27)
			break;
	}
	//clean up
	cv::destroyAllWindows();
	return 0;
}
/********************************************************************************************************************************/
//camera records and detects human faces
int recording(cv::Mat frame, cv::VideoCapture real_time) {
	//front face detector
	std::string frontalface_alt_classifier_location = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	cv::CascadeClassifier face_detector;
	if (!face_detector.load(frontalface_alt_classifier_location)) {
		std::cout << "ERROR: cannot load frontal face classifier from location" << std::endl;
		exit(1);
	}

	//profile face detector (side of face)
	std::string profileface_classifier_location = "D:/opencv/sources/data/haarcascades/haarcascade_profileface.xml";
	cv::CascadeClassifier profileface_detector;
	if (!profileface_detector.load(profileface_classifier_location)) {
		std::cout << "ERROR: cannot load profile face classifier from location" << std::endl;
		exit(1);
	}

	//full body detector
	std::string fullbody_classifier_location = "D:/opencv/sources/data/haarcascades/haarcascade_fullbody.xml";
	cv::CascadeClassifier body_detector;
	if (!body_detector.load(fullbody_classifier_location)) {
		std::cout << "ERROR: cannot load full body classifier from location" << std::endl;
		exit(1);
	}

	//storing video file
	cv::VideoWriter Recording("D:/Projects/Tracking/output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 17, cv::Size(cv::CAP_PROP_FRAME_WIDTH, cv::CAP_PROP_FRAME_HEIGHT));

	//vars that store what the camera sees
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> profilefaces;
	std::vector<cv::Rect> bodies;

	//picture variables
	std::vector<int> pic_params;
	pic_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	pic_params.push_back(3);

	//picture vars for save location
	std::stringstream face_pic_path_name;
	std::stringstream profile_pic_path_name;
	std::stringstream body_pic_path_name;
	std::string file_type = ".png";
	int body_pic_num = 0;
	int profile_pic_num = 0;
	int face_pic_num = 0;

	//counting vars
	int num_of_faces = 0;
	int num_of_profiles = 0;
	int num_of_people = 0;

	//////////////////////////////////////////////////////////////////////////////////
	//MAIN LOOP
	while (true) {
		//get an image from the camera
		real_time.read(frame);

		//if camera is interrupted, will break from loop and await reconnection
		//if(real_time.read(frame)){
		//	//get time of disconnect
		//	std::cout << "ERROR: Camera disconnected at " << time(NULL) << ". Waiting for reconnect..." << std::endl;
		//	break;
		//}

		//do we detect something?
		face_detector.detectMultiScale(frame, faces, 1.1, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
		profileface_detector.detectMultiScale(frame, profilefaces, 1.1, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
		body_detector.detectMultiScale(frame, bodies, 1.1, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(75, 100));//may need to change Size() of bodies we detect

		//need to change specs so the size

		//highlight detected faces
		for (int i = 0; i < faces.size(); i++) {
			cv::ellipse(frame, cv::Point(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5), cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 1, 8, 0);
		}

		//highlight detected faces
		for (int i = 0; i < profilefaces.size(); i++) {
			cv::ellipse(frame, cv::Point(profilefaces[i].x + profilefaces[i].width * 0.5, profilefaces[i].y + profilefaces[i].height * 0.5), cv::Size(profilefaces[i].width * 0.5, profilefaces[i].height * 0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 1, 8, 0);
		}

		//highlight detected bodies
		for (int j = 0; j < bodies.size(); j++) {
			cv::rectangle(frame, cv::Point(bodies[j].x, bodies[j].y), cv::Point(bodies[j].x + bodies[j].width, bodies[j].y + bodies[j].height), cv::Scalar(255, 0, 255), 1, 8, 0);
		}

		//Can probably turn these into one if statement.

		//if there is a new number of faces, dont want to take a picture every frame
		if (num_of_faces < faces.size()) {
			face_pic_path_name << "D:/Projects/Tracking/face_pic_images/face_pic_image" << face_pic_num << file_type;
			cv::imwrite(face_pic_path_name.str(), frame, pic_params);
			face_pic_path_name.str(std::string());//using empty string* to technically be a little faster
			face_pic_num++;
		}

		//save image of new profile
		if (num_of_profiles < profilefaces.size()) {
			profile_pic_path_name << "D:/Projects/Tracking/face_pic_images/profile_pic_image" << profile_pic_num << file_type;
			cv::imwrite(profile_pic_path_name.str(), frame, pic_params);
			profile_pic_path_name.str(std::string());
			profile_pic_num++;
		}

		//if there is a new number of bodies/people, dont want to take a picture every frame
		if (num_of_people < bodies.size()) {
			body_pic_path_name << "D:/Projects/Tracking/body_pic_images/body_pic_image" << body_pic_num << file_type;
			cv::imwrite(body_pic_path_name.str(), frame, pic_params);
			face_pic_path_name.str(std::string());
			body_pic_num++;
		}

		//TODO
		//create database of pictures and in another program (maybe parallel process) try to identify if the person that is detected is someone familiar
		//need a machine to learn features of people who come here a lot.
		//machine gets smarter because it adds another picture to the list of what it recognizes as a specific person

		//compare values next iteration, see if we have a new feature to capture.
		num_of_faces = faces.size();
		num_of_profiles = profilefaces.size();
		num_of_people = bodies.size();

		//save frame in the video recording
		Recording.write(frame);

		//show everything
		cv::imshow("Face Detection", frame);

		if (cv::waitKey(10) == 27)
			cv::destroyAllWindows();
		return -1;
	}
	printf("STOPPED RECORDING!\n");

	//dont want to destroy all windows bc want to see last frame before disconnect, could have clues
	//cv::destroyAllWindows();
	return 0;

}

/********************************************************************************************************************************/
//basically a do nothing loop, keeps checking to see if camera is detected
int camera_disconnected(cv::Mat frame, cv::VideoCapture real_time) {
	std::cout << "Camera Disconnected: waiting for camera to be reconnected..." << std::endl;

	//check for new camera/ reconnected camera
	while (true) {
		//need to modify the same var and pass it to other fuctions for security reasons. If the camera is reconnected but is disconnected when the program is beginning to
		//execute recording() causes the program to crash/stop when trying to record from a camera that doesn't exist but it thinks exists.
		//already taken care of: in recording() it checks at the beginning of every loop

		//make sure camera input isn't interrupted as we read from input
		if (real_time.read(frame)) {
			//get time of reconnect
			std::cout << "Camera reconnected at " << time(NULL) << std::endl;
			return 0;
		}
	}
}
