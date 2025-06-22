#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TransformStamped.h>

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "../include/GpsTypes.h"

#include"../../../include/System.h"
#include"../include/GpsTypes.h"

using namespace std;

class GpsGrabber
{
public:
    GpsGrabber(bool bUseGps, double _noiseStdDev)
      : mbUseGps(bUseGps), noiseStdDev(_noiseStdDev), gpsMeasId(0)
    {
        const double mean = 0.0;
        std::random_device rd;
        generator = std::mt19937(rd());
        noiseX = std::normal_distribution<float>(mean, noiseStdDev);
        noiseY = std::normal_distribution<float>(mean, noiseStdDev);
        noiseZ = std::normal_distribution<float>(mean, noiseStdDev);
    }

    void GrabGps(const sensor_msgs::NavSatFixConstPtr &gps_msg);
    void SaveMeasurement(const ORB_SLAM3::GlobalPosition::GlobalPosition*);
    void ClearMeasurements();

    bool mbUseGps;
    queue<const ORB_SLAM3::GlobalPosition::GlobalPosition*> gpsBuf;
    list<const ORB_SLAM3::GlobalPosition::GlobalPosition*> allMeasurements;
    mutex mBufMutex;    
    double noiseStdDev;
    int gpsMeasId;
    normal_distribution<float> noiseX, noiseY, noiseZ;
    mt19937 generator;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, GpsGrabber *pGpsGb, bool bRect, bool bClahe)
      : mpSLAM(pSLAM), mpGpsGb(pGpsGb), do_rectify(bRect), mbClahe(bClahe)
    {
        if(mbClahe)
            mClahe = cv::createCLAHE(3.0, cv::Size(8,8));
    }

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncLoop();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    mutex mBufMutexLeft, mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    GpsGrabber *mpGpsGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe;
};

bool useGps(const string &strSettingsFile)
{
    cv::FileStorage fs(strSettingsFile, cv::FileStorage::READ);
    if(!fs.isOpened()) { cerr<<"ERROR: cannot open "<<strSettingsFile<<endl; throw -1; }
    int u = fs["System.UseGlobalMeas"];
    return u!=0;
}

float getNoiseStdDev(const string &strSettingsFile)
{
    cv::FileStorage fs(strSettingsFile, cv::FileStorage::READ);
    if(!fs.isOpened()) { cerr<<"ERROR: cannot open "<<strSettingsFile<<endl; throw -1; }
    return fs["System.GPSSimulatedNoise"];
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "GNSS_Stereo");
    ros::NodeHandle n("~");
    if(argc < 4 || argc > 5) {
        cerr<<"\nUsage: rosrun ORB_SLAM3 Stereo_Only VOC_PATH SETTINGS_PATH do_rectify [do_equalize]\n";
        return 1;
    }
    bool bEqual = (argc==5 && string(argv[4])=="true");
    bool bRect  = (string(argv[3])=="true");

    // *** PURE STEREO MODE – no IMU ***
    ORB_SLAM3::System SLAM(argv[1], argv[2],
                           ORB_SLAM3::System::STEREO_GPS,
                           true);

    bool mbUseGps     = useGps(argv[2]);
    float noiseStdDev = getNoiseStdDev(argv[2]);
    GpsGrabber gpsgb(mbUseGps, noiseStdDev);
    ImageGrabber igb(&SLAM, &gpsgb, bRect, bEqual);

    if(bRect) {
        cv::FileStorage fs(argv[2], cv::FileStorage::READ);
        cv::Mat Kl,Kr,Pl,Pr,Rl,Rr,Dl,Dr;
        fs["LEFT.K"] >> Kl; fs["RIGHT.K"] >> Kr;
        fs["LEFT.P"] >> Pl; fs["RIGHT.P"] >> Pr;
        fs["LEFT.R"] >> Rl; fs["RIGHT.R"] >> Rr;
        fs["LEFT.D"] >> Dl; fs["RIGHT.D"] >> Dr;
        int rl=fs["LEFT.height"], cl=fs["LEFT.width"];
        cv::initUndistortRectifyMap(Kl,Dl,Rl,Pl.rowRange(0,3).colRange(0,3),
                                    cv::Size(cl,rl),CV_32F,
                                    igb.M1l, igb.M2l);
        int rr=fs["RIGHT.height"], cr=fs["RIGHT.width"];
        cv::initUndistortRectifyMap(Kr,Dr,Rr,Pr.rowRange(0,3).colRange(0,3),
                                    cv::Size(cr,rr),CV_32F,
                                    igb.M1r, igb.M2r);
    }

    ros::Subscriber sub_l = n.subscribe("/camera/left/image_raw",  100,
                                        &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_r = n.subscribe("/camera/right/image_raw", 100,
                                        &ImageGrabber::GrabImageRight,&igb);
    ros::Subscriber sub_g = n.subscribe("/gps/fix", 1000,
                                        &GpsGrabber::GrabGps, &gpsgb);

    thread sync_thread(&ImageGrabber::SyncLoop, &igb);

    ros::spin();
    if(!ros::ok())
    {
        std::cout << "Shutting down..." << std::endl;
        SLAM.SaveGeographicalTrajectory("CameraTrajectoryGeographical.txt");
        SLAM.forceSystemShutdown(false);
        gpsgb.ClearMeasurements();
    }

    return 0;
}

void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &msg)
{
    lock_guard<mutex> lk(mBufMutexLeft);
    if(!imgLeftBuf.empty()) imgLeftBuf.pop();
    imgLeftBuf.push(msg);
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &msg)
{
    lock_guard<mutex> lk(mBufMutexRight);
    if(!imgRightBuf.empty()) imgRightBuf.pop();
    imgRightBuf.push(msg);
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img_msg,
                 sensor_msgs::image_encodings::MONO8);
    } catch(cv_bridge::Exception &e) {
        ROS_ERROR("%s", e.what());
        return cv::Mat();
    }
    return cv_ptr->image.clone();
}

void ImageGrabber::SyncLoop()
{
    const double maxDiff = 0.01;
    while(ros::ok())
    {
        if(imgLeftBuf.empty() || imgRightBuf.empty())
        {
            this_thread::sleep_for(1ms);
            continue;
        }

        // timestamp sync
        double tL = imgLeftBuf.front()->header.stamp.toSec();
        double tR = imgRightBuf.front()->header.stamp.toSec();
        {
            lock_guard<mutex> lkR(mBufMutexRight);
            while((tL - tR) > maxDiff && imgRightBuf.size()>1) {
                imgRightBuf.pop();
                tR = imgRightBuf.front()->header.stamp.toSec();
            }
        }
        {
            lock_guard<mutex> lkL(mBufMutexLeft);
            while((tR - tL) > maxDiff && imgLeftBuf.size()>1) {
                imgLeftBuf.pop();
                tL = imgLeftBuf.front()->header.stamp.toSec();
            }
        }
        if(fabs(tL - tR) > maxDiff) continue;

        // get images
        cv::Mat imL, imR;
        {
            lock_guard<mutex> lk(mBufMutexLeft);
            imL = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();
        }
        {
            lock_guard<mutex> lk(mBufMutexRight);
            imR = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
        }

        if(mbClahe) {
            mClahe->apply(imL, imL);
            mClahe->apply(imR, imR);
        }
        if(do_rectify) {
            cv::remap(imL, imL, M1l, M2l, cv::INTER_LINEAR);
            cv::remap(imR, imR, M1r, M2r, cv::INTER_LINEAR);
        }

        // GPS (optional)
        vector<const ORB_SLAM3::GlobalPosition::GlobalPosition*> vg;
        if(mpGpsGb->mbUseGps) {
            lock_guard<mutex> lk(mpGpsGb->mBufMutex);
            while(!mpGpsGb->gpsBuf.empty() &&
                  mpGpsGb->gpsBuf.front()->timestamp <= tL)
            {
                vg.push_back(mpGpsGb->gpsBuf.front());
                mpGpsGb->gpsBuf.pop();
            }
        }
        
        // std::cout<< "start feeding images and gps data" << std::endl;
        // std::cout<< "gps measurement size: " << vg.size() << std::endl;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        mpSLAM->TrackStereo(imL, imR, tL, vImuMeas, vg);
        // std::cout<<"finish feeding images and gps data" << std::endl;

        this_thread::sleep_for(1ms);
    }
}

void GpsGrabber::GrabGps(const sensor_msgs::NavSatFixConstPtr &gps_msg)
{
    if(!mbUseGps) return;
    Eigen::Matrix3d cov; cov.setZero();
    if(noiseStdDev==0.0){
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)
            cov(i,j) = gps_msg->position_covariance[3*i+j];
    } else {
        for(int i=0;i<3;i++) cov(i,i)=noiseStdDev*noiseStdDev;
    }
    Eigen::Vector3d noise(
        noiseX(generator),
        noiseY(generator),
        noiseZ(generator)
    );
    auto *m = new ORB_SLAM3::GlobalPosition::GpsMeasurement(
        gpsMeasId++,
        gps_msg->latitude,
        gps_msg->longitude,
        gps_msg->altitude,
        gps_msg->header.stamp.toSec(),
        cov,
        noiseStdDev>0.0 ? noise : Eigen::Vector3d(0,0,0)
    );
    {
        lock_guard<mutex> lk(mBufMutex);
        gpsBuf.push(m);
    }
    allMeasurements.push_back(m);
}

void GpsGrabber::ClearMeasurements()
{
    for(auto m: allMeasurements) delete m;
}
