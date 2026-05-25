// Incrementally accumulates a Burt-Adelson Laplacian pyramid blend across batches of images.
// For each input image, builds its Laplacian pyramid and accumulates per-level weighted sums.
// Focus weight at each level is the smoothed local energy (magnitude) of the Laplacian coefficients.
//
// Reference: P. Burt and E. Adelson, "A multiresolution spline with application to image mosaics",
//            ACM Transactions on Graphics, 1983.

#pragma once
#include "worker.hh"
#include <vector>

namespace focusstack {

class Task_PyramidMerge: public Task
{
public:
  // prev_merge: result from previous batch, or nullptr for first batch.
  // images: batch of aligned color (or grayscale) ImgTask results.
  Task_PyramidMerge(std::shared_ptr<Task_PyramidMerge> prev_merge,
                    const std::vector<std::shared_ptr<ImgTask>>& images);

  // Accumulated weighted Laplacian pyramid (one Mat per level, CV_32FC3 or CV_32FC1).
  const std::vector<cv::Mat>& num_pyramid() const { return m_num; }

  // Accumulated weight pyramid (one Mat per level, CV_32FC1).
  const std::vector<cv::Mat>& denom_pyramid() const { return m_denom; }

  int levels() const { return static_cast<int>(m_num.size()); }
  cv::Rect valid_area() const { return m_valid_area; }

  // Number of pyramid levels chosen for a given image size.
  static int levels_for_size(cv::Size size);

  // Build the Laplacian pyramid for one image into out_lap.
  // The base (coarsest) level is the Gaussian remainder.
  static void build_laplacian(const cv::Mat& img, std::vector<cv::Mat>& out_lap, int levels);

  // Compute per-level focus weight map (smoothed squared magnitude of Laplacian).
  static void focus_weights(const std::vector<cv::Mat>& lap, std::vector<cv::Mat>& out_w);

private:
  virtual void task();

  std::shared_ptr<Task_PyramidMerge> m_prev;
  std::vector<std::shared_ptr<ImgTask>> m_images;

  std::vector<cv::Mat> m_num;   // sum of (weight * laplacian) per level
  std::vector<cv::Mat> m_denom; // sum of weights per level
  cv::Rect m_valid_area;
};

}
