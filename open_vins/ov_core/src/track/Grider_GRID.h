/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_GRIDER_GRID_H
#define OV_CORE_GRIDER_GRID_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/opencv_lambda_body.h"

namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in each.
 * We then pick enough top points in each grid so that we have the total number of desired points.
 */
class Grider_GRID {

public:
  /**
   * @brief Compare keypoints based on their response value.
   * @param first First keypoint
   * @param second Second keypoint
   *
   * We want to have the keypoints with the highest values!
   * See: https://stackoverflow.com/a/10910921
   */
  static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) { return first.response > second.response; }

  /**
   * @brief This function will perform grid extraction using FAST.
   * @param img Image we will do FAST extraction on
   * @param mask Region of the image we do not want to extract features in (255 = do not detect features)
   * @param valid_locs Valid 2d grid locations we will extract in (instead of the whole image)
   * @param pts vector of extracted points we will return
   * @param num_features max number of features we want to extract
   * @param grid_x size of grid in the x-direction / u-direction
   * @param grid_y size of grid in the y-direction / v-direction
   * @param threshold FAST threshold paramter (10 is a good value normally)
   * @param nonmaxSuppression if FAST should perform non-max suppression (true normally)
   *
   * Given a specified grid size, this will try to extract fast features from each grid.
   * It will then return the best from each grid in the return vector.
   */
  static void perform_griding(const cv::Mat &img,
                              const cv::Mat &mask,
                              const std::vector<std::pair<int,int>> &valid_locs,
                              std::vector<cv::KeyPoint> &pts,
                              int num_features,
                              int grid_x, int grid_y,
                              int threshold,
                              bool nonmaxSuppression)
  {
    if (valid_locs.empty()) return;

    /* ---------- 1. 重新调整 grid ---------- */
    if (num_features < grid_x * grid_y) {
      double r = static_cast<double>(grid_x) / grid_y;
      grid_y   = static_cast<int>(std::ceil(std::sqrt(num_features / r)));
      grid_x   = static_cast<int>(std::ceil(grid_y * r));
    }
    const int num_features_grid = static_cast<int>( (double)num_features / (grid_x*grid_y) ) + 1;

    const int size_x = img.cols / grid_x;
    const int size_y = img.rows / grid_y;
    CV_Assert(size_x>0 && size_y>0);

    /* ---------- 2. 统计变量 ---------- */
    std::atomic<size_t> fast_total{0};
    std::atomic<size_t> nonempty_buckets{0};

    /* ---------- 3. 并行 FAST ---------- */
    std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());
    parallel_for_(cv::Range(0,(int)valid_locs.size()),LambdaBody([&](const cv::Range &range){
      for(int r=range.start;r<range.end;++r){
        auto cell = valid_locs[r];
        int x = cell.first  * size_x;
        int y = cell.second * size_y;
        if(x+size_x>img.cols||y+size_y>img.rows) continue;

        cv::Rect roi(x,y,size_x,size_y);
        std::vector<cv::KeyPoint> cand;  // 候选
        cv::FAST(img(roi), cand, threshold, nonmaxSuppression);

        if(!cand.empty()){
          nonempty_buckets.fetch_add(1, std::memory_order_relaxed);
          fast_total.fetch_add(cand.size(), std::memory_order_relaxed);
        }

        std::sort(cand.begin(), cand.end(),
                  [](const cv::KeyPoint&a,const cv::KeyPoint&b){return a.response>b.response;});

        for(size_t i=0;i< (size_t)num_features_grid && i<cand.size(); ++i){
          cv::KeyPoint kp = cand[i];
          kp.pt.x += static_cast<float>(x);
          kp.pt.y += static_cast<float>(y);
          if(kp.pt.x<0||kp.pt.x>=img.cols||kp.pt.y<0||kp.pt.y>=img.rows) continue;
          if(mask.at<uint8_t>((int)kp.pt.y,(int)kp.pt.x)>127) continue;
          collection[r].push_back(kp);
        }
      }
    }));

    /* ---------- 4. 汇总 ---------- */
    pts.clear();
    for(auto &vec:collection) pts.insert(pts.end(), vec.begin(), vec.end());
    if(pts.empty()) return;

    /* ---------- 5. Sub‑pixel 精化 ---------- */
    std::vector<cv::Point2f> subpix; subpix.reserve(pts.size());
    for(auto &k:pts) subpix.emplace_back(k.pt);
    cv::cornerSubPix(img, subpix, cv::Size(5,5), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.001));
    for(size_t i=0;i<pts.size();++i) pts[i].pt=subpix[i];

    /* ---------- 6. 调试打印 ---------- */
    std::cout << "[CPU-GRIDER] FAST候选=" << fast_total.load()
              << ", 非空bucket="          << nonempty_buckets.load()
              << ", 最终保留="            << pts.size() << std::endl;
  }
};

}  // namespace ov_core

#endif // OV_CORE_GRIDER_GRID_H

