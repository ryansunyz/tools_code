#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<vector>

struct PCLPointXYZITB {
    float x;
    float y;
    float z;
    uint8_t intensity;
    float timestamp;
    uint32_t beam_id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    PCLPointXYZITB,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (uint8_t, intensity, intensity)
    (float, timestamp, timestamp)
    (uint32_t, beam_id, beam_id)
)

std::vector<std::string> pcd_list_;

bool LoadFileList(std::string& pcd_dir) {
    namespace fs  =  boost::filesystem;
    fs::path full_path(pcd_dir);
    if (!fs::exists(full_path)) {
        return false;
    }

    fs::directory_iterator item_begin(full_path);
    fs::directory_iterator item_end;
    for (;item_begin != item_end; item_begin++) {
        if (fs::is_directory(*item_begin)) {
            std::cout << item_begin->path().native() << "\n";
            continue;
        } else {
            std::string file_name = item_begin->path().native();
            if (file_name.substr(file_name.size() - 3, 3) == "pcd") {
                pcd_list_.push_back(item_begin->path().native());
            }
        }
    }
    std::sort(pcd_list_.begin(), pcd_list_.end());
    std::cout << "PCD directory: " << pcd_dir << ", size: " << pcd_list_.size() << std::endl;
    return true;
}

int ProcessIntensity(std::string& filepath) {
    //创建了一个名为cloud的指针，储存XYZ类型的点云数据
     pcl::PointCloud<PCLPointXYZITB>::Ptr cloud(new pcl::PointCloud<PCLPointXYZITB>);
     //*打开点云文件

    if (pcl::io::loadPCDFile<PCLPointXYZITB>(filepath, *cloud) == -1) {
        PCL_ERROR("Couldn't read file rabbit.pcd\n");
        return(-1);
    }
    //
    pcl::PointCloud<PCLPointXYZITB>::Ptr cloudout(new pcl::PointCloud<PCLPointXYZITB>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        PCLPointXYZITB p;
        if (cloud->points[i].beam_id < 5) {
            cloudout->push_back(p);
        } else {
            continue;
        }
    }
    std::string savename = filepath.substr(0, filepath.size()-4) + "_new.pcd";
    if (!cloudout->empty()) {
        pcl::io::savePCDFileBinary(savename, *cloudout);
        std::cout << " save pointcloud size = " << cloudout->size() << std::endl;
    }
    std::cout << "Loaded:" << cloud->width*cloud->height<<"data points from test_pcd.pcd with the following fields:"<< std::endl;
    return 0;
}
int main(int argc, char** argv)
{
    // std::string filepath = "/home/syz/Public/pcdread/test.pcd";
    std::string data_path = "/home/syz/Public/pcdread/pcd";
    LoadFileList(data_path);
    for(size_t i = 0; i < pcd_list_.size(); ++i) {
        std::cout << "Pocessing file -> " << pcd_list_[i] << std::endl;
        ProcessIntensity(pcd_list_[i]);
    }
}