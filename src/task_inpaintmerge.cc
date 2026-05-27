#include "task_inpaintmerge.hh"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using namespace focusstack;

Task_InpaintMerge::Task_InpaintMerge(std::shared_ptr<Task_Merge> merge, float threshold):
  m_merge(merge), m_threshold(threshold)
{
  m_name     = "Inpaint unfocused regions";
  m_filename = "inpainted_" + merge->filename();
  m_depends_on.push_back(merge);
}

void Task_InpaintMerge::fill_holes(const cv::Mat& input, cv::Mat& output,
                                   const cv::Mat& valid_mask, int radius)
{
  // Build a weight image: 1 where valid, 0 in holes.
  cv::Mat weight(input.size(), CV_32F, cv::Scalar(0));
  weight.setTo(1.0f, valid_mask);

  // Zero out hole pixels in a copy of the image so they don't contribute
  // to the Gaussian average.
  cv::Mat img_masked;
  input.copyTo(img_masked);
  img_masked.setTo(0.0f, ~valid_mask);

  int ksize = radius * 4 + 1;
  cv::Mat blurred_w, blurred_img;
  cv::GaussianBlur(weight,     blurred_w,   cv::Size(ksize, ksize), radius);
  cv::GaussianBlur(img_masked, blurred_img, cv::Size(ksize, ksize), radius);

  // Divide to get the weighted average; avoid division by near-zero.
  cv::Mat filled;
  cv::divide(blurred_img, cv::max(blurred_w, 1e-6f), filled);

  // Copy the filled values only into hole positions; confident pixels keep
  // their original values.
  output = input.clone();
  filled.copyTo(output, ~valid_mask);
}

void Task_InpaintMerge::task()
{
  const cv::Mat& merged      = m_merge->img();
  const cv::Mat& max_sq      = m_merge->max_absval(); // squared magnitudes
  m_valid_area               = m_merge->valid_area();

  if (merged.empty() || max_sq.empty())
    throw std::runtime_error("Task_InpaintMerge: empty input from Task_Merge");

  // Build the confidence mask.
  // max_sq holds squared wavelet magnitudes; compare to threshold squared.
  float threshold_sq = m_threshold * m_threshold;
  cv::Mat valid_mask = (max_sq >= threshold_sq); // CV_8UC1: 255 = confident

  // Count holes for logging.
  int hole_count = cv::countNonZero(~valid_mask);
  if (m_logger)
    m_logger->verbose("InpaintMerge: %d hole pixels (%.1f%%) below focus threshold %.1f\n",
                      hole_count,
                      100.0f * hole_count / (merged.rows * merged.cols),
                      m_threshold);

  if (hole_count == 0)
  {
    // Nothing to do — just forward the result.
    m_result = merged.clone();
    m_merge.reset();
    return;
  }

  // Use a fill radius large enough to bridge typical focus-gap widths.
  // Capped at 1/8 of the shorter image dimension so it stays tractable.
  int radius = std::max(64, std::min(merged.rows, merged.cols) / 8);

  // Process the complex wavelet image channel by channel (real + imaginary).
  std::vector<cv::Mat> channels;
  cv::split(merged, channels);

  for (auto& ch : channels)
  {
    cv::Mat filled;
    fill_holes(ch, filled, valid_mask, radius);
    ch = filled;
  }

  cv::merge(channels, m_result);
  m_merge.reset();
}
