/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
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

#ifndef OV_CORE_GRIDER_GPU_H
#define OV_CORE_GRIDER_GPU_H

#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

#include <algorithm>
#include <vector>

namespace ov_core {

class Grider_GPU {
public:
  static void perform_griding(const cv::Mat &img, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
                              std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
                              bool nonmaxSuppression) {
    if (img.empty() || valid_locs.empty())
      return;

    if (num_features < grid_x * grid_y) {
      double r = static_cast<double>(grid_x) / grid_y;
      grid_y = static_cast<int>(std::ceil(std::sqrt(num_features / r)));
      grid_x = static_cast<int>(std::ceil(grid_y * r));
    }
    const int num_features_grid = static_cast<int>((double)num_features / (grid_x * grid_y)) + 1;

    const int size_x = img.cols / grid_x;
    const int size_y = img.rows / grid_y;
    CV_Assert(size_x > 0 && size_y > 0);

#ifdef HAVE_OPENCV_CUDAFEATURES2D
    cv::cuda::GpuMat img_gpu(img);
    auto detector = cv::cuda::FastFeatureDetector::create(threshold, nonmaxSuppression);
#endif

    std::vector<cv::KeyPoint> collected;
    for (const auto &cell : valid_locs) {
      int x = cell.first * size_x;
      int y = cell.second * size_y;
      if (x + size_x > img.cols || y + size_y > img.rows)
        continue;
      cv::Rect roi(x, y, size_x, size_y);
      std::vector<cv::KeyPoint> cand;
#ifdef HAVE_OPENCV_CUDAFEATURES2D
      cv::cuda::GpuMat roi_gpu(img_gpu, roi);
      detector->detect(roi_gpu, cand);
#else
      cv::FAST(img(roi), cand, threshold, nonmaxSuppression);
#endif
      if (cand.empty())
        continue;
      std::sort(cand.begin(), cand.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
      for (size_t i = 0; i < (size_t)num_features_grid && i < cand.size(); ++i) {
        cv::KeyPoint kp = cand[i];
        kp.pt.x += static_cast<float>(x);
        kp.pt.y += static_cast<float>(y);
        if (kp.pt.x < 0 || kp.pt.x >= img.cols || kp.pt.y < 0 || kp.pt.y >= img.rows)
          continue;
        if (mask.at<uint8_t>((int)kp.pt.y, (int)kp.pt.x) > 127)
          continue;
        collected.push_back(kp);
      }
    }

    if (collected.empty())
      return;

    cv::Size win_size(5, 5);
    cv::Size zero_zone(-1, -1);
    cv::TermCriteria term_crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.001);
    std::vector<cv::Point2f> subpix;
    subpix.reserve(collected.size());
    for (auto &k : collected)
      subpix.emplace_back(k.pt);
    cv::cornerSubPix(img, subpix, win_size, zero_zone, term_crit);
    for (size_t i = 0; i < collected.size(); ++i)
      collected[i].pt = subpix[i];

    std::sort(collected.begin(), collected.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
    if ((int)collected.size() > num_features)
      collected.resize(num_features);
    pts = std::move(collected);
  }
};

} // namespace ov_core

#endif // OV_CORE_GRIDER_GPU_H
