#include "task_saveimg.hh"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace focusstack;

Task_SaveImg::Task_SaveImg(std::string filename, std::shared_ptr<ImgTask> input, std::shared_ptr<ImgTask> alphamask,
  int jpgquality, bool nocrop, bool normalize)
{
  m_filename = filename;

  if (filename != "" && filename != ":memory:")
  {
    m_name = "Save " + filename;
  }
  else
  {
    m_name = "Final crop " + input->filename();
  }

  m_jpgquality = jpgquality;
  m_nocrop = nocrop;
  m_normalize = normalize;
  m_input = input;
  m_depends_on.push_back(input);

  if (alphamask)
  {
    m_alphamask = alphamask;
    m_depends_on.push_back(alphamask);
  }
}

void Task_SaveImg::task()
{
  if (m_nocrop)
  {
    m_result = m_input->img();
    m_valid_area = m_input->valid_area();
  }
  else
  {
    cv::Size origsize = m_input->img().size();
    m_result = m_input->img_cropped();
    m_valid_area = cv::Rect(0, 0, m_result.cols, m_result.rows);

    if (origsize != m_result.size())
    {
      m_logger->verbose("%s cropped from (%d, %d) to (%d, %d)\n",
            m_filename.c_str(), origsize.width, origsize.height, m_result.cols, m_result.rows);
    }
  }

  if (m_result.channels() == 2)
  {
    // Convert complex wavelets to RGB image
    cv::Mat channels[3];
    cv::split(m_result, channels);
    channels[0].convertTo(channels[0], CV_8U);
    channels[1].convertTo(channels[1], CV_8U);
    channels[2].create(m_result.rows, m_result.cols, CV_8U);
    channels[2] = 0;
    cv::merge(channels, 3, m_result);
  }

  // Per-channel linear stretch to full 0..255 range
  if (m_normalize && m_result.type() == CV_8UC3)
  {
    cv::Mat channels[3];
    cv::split(m_result, channels);
    for (int c = 0; c < 3; c++)
    {
      cv::normalize(channels[c], channels[c], 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    cv::merge(channels, 3, m_result);
    m_logger->verbose("Normalized output to full color range\n");
  }
  else if (m_normalize && m_result.type() == CV_8U)
  {
    cv::normalize(m_result, m_result, 0, 255, cv::NORM_MINMAX, CV_8U);
    m_logger->verbose("Normalized grayscale output to full range\n");
  }

  // Add alpha channel if given
  if (m_alphamask)
  {
    cv::Mat channels[4];

    if (m_result.channels() == 1)
    {
      channels[0] = channels[1] = channels[2] = m_result;
    }
    else
    {
      cv::split(m_result, channels);
    }

    if (m_nocrop)
    {
      channels[3] = m_alphamask->img();
    }
    else
    {
      channels[3] = m_alphamask->img_cropped();
    }

    cv::merge(channels, 4, m_result);
  }

  // Input image can be released now
  m_input.reset();
  m_alphamask.reset();

  if (m_filename != "" && m_filename != ":memory:")
  {
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(m_jpgquality);
    cv::imwrite(m_filename, m_result, compression_params);
  }
}


