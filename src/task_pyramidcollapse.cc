#include "task_pyramidcollapse.hh"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using namespace focusstack;

Task_PyramidCollapse::Task_PyramidCollapse(std::shared_ptr<Task_PyramidMerge> merged):
  m_merged(merged)
{
  m_name = "PyramidCollapse";
  m_filename = "pyramid_result.jpg";
  m_depends_on.push_back(merged);
}

void Task_PyramidCollapse::task()
{
  const std::vector<cv::Mat>& num   = m_merged->num_pyramid();
  const std::vector<cv::Mat>& denom = m_merged->denom_pyramid();

  if (num.empty())
    throw std::runtime_error("Task_PyramidCollapse: empty pyramid");

  int levels = static_cast<int>(num.size());

  // Normalize each level: where denom > 0, divide; elsewhere use 0.
  std::vector<cv::Mat> blended(levels);
  for (int l = 0; l < levels; l++)
  {
    cv::Mat mask = denom[l] > 0;
    blended[l] = cv::Mat::zeros(num[l].size(), num[l].type());

    if (num[l].channels() == 1)
    {
      cv::divide(num[l], denom[l], blended[l]);
      blended[l].setTo(0, ~mask);
    }
    else
    {
      // Broadcast scalar denom across color channels.
      std::vector<cv::Mat> num_chans;
      cv::split(num[l], num_chans);
      std::vector<cv::Mat> result_chans;
      for (auto& c : num_chans)
      {
        cv::Mat out;
        cv::divide(c, denom[l], out);
        out.setTo(0, ~mask);
        result_chans.push_back(out);
      }
      cv::merge(result_chans, blended[l]);
    }
  }

  // Collapse: start from coarsest level and work towards finest.
  cv::Mat result = blended[levels - 1].clone();
  for (int l = levels - 2; l >= 0; l--)
  {
    cv::Mat up;
    cv::pyrUp(result, up, blended[l].size());
    result = up + blended[l];
  }

  // Clip to [0, 255] and convert to 8-bit.
  cv::Mat clipped;
  cv::threshold(result,  clipped, 255.0, 255.0, cv::THRESH_TRUNC);
  cv::threshold(clipped, clipped,   0.0,   0.0, cv::THRESH_TOZERO);
  clipped.convertTo(m_result, CV_8U);

  m_valid_area = m_merged->valid_area();

  m_merged.reset();
}
