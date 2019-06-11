#include <ros/ros.h>
#include <test2/myservice.h>
#include <string>

bool handle_fun(test2::myservice::Request &req, test2::myservice::Response &res)
{
    res.sum = req.a+req.b;
    res.feedback = "My answer is " + std::to_string(res.sum);
    return true;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "test2_server");
    ros::NodeHandle n;
    ros::ServiceServer service = n.advertiseService("mytestservice", handle_fun);
    ROS_INFO("server start");
    ros::spin();    
    return 0;
}


