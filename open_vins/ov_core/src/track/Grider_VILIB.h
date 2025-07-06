#ifndef OV_CORE_GRIDER_VILIB_H
#define OV_CORE_GRIDER_VILIB_H

#include <opencv2/opencv.hpp>
#include <vilib/common/frame.h>
#include <vilib/feature_detection/fast/fast_gpu.h>
#include <vilib/storage/pyramid_pool.h>
#include <vilib/config.h>

namespace ov_core {

class Grider_VILIB {
public:
  static void perform_griding(const cv::Mat &img,
                              const cv::Mat &mask,
                              const std::vector<std::pair<int,int>> &valid_locs,
                              std::vector<cv::KeyPoint> &pts,
                              int num_features,
                              int grid_x,
                              int grid_y,
                              int threshold,
                              bool nonmaxSuppression)
  {
    if (img.empty() || valid_locs.empty()) return;

    vilib::PyramidPool::init(1,
                             img.cols,
                             img.rows,
                             1,
                             vilib::MAX_IMAGE_PYRAMID_LEVELS,
                             vilib::IMAGE_PYRAMID_MEMORY_TYPE);

    vilib::Frame frame(img, 0, vilib::MAX_IMAGE_PYRAMID_LEVELS);

    vilib::FASTGPU detector(img.cols,
                            img.rows,
                            img.cols / grid_x,
                            img.rows / grid_y,
                            0,
                            1,
                            0,
                            0,
                            static_cast<float>(threshold),
                            10,
                            vilib::fast_score::SUM_OF_ABS_DIFF_ON_ARC);

    detector.detect(frame.pyramid_);
    const auto &dpts = detector.getPoints();
    for (const auto &p : dpts) {
      if (p.level_ != 0) continue;
      int x = static_cast<int>(p.x_);
      int y = static_cast<int>(p.y_);
      if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) continue;
      if (mask.at<uint8_t>(y, x) > 127) continue;
      cv::KeyPoint kp;
      kp.pt.x = p.x_;
      kp.pt.y = p.y_;
      kp.response = p.score_;
      kp.octave = p.level_;
      pts.push_back(kp);
      if ((int)pts.size() >= num_features) break;
    }

    vilib::PyramidPool::deinit();
  }
};

} // namespace ov_core

#endif // OV_CORE_GRIDER_VILIB_H
