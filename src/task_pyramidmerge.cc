#include "task_pyramidmerge.hh"
#include "task_wavelet.hh"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using namespace focusstack;

// Radius (in pixels) of the Gaussian used to smooth focus weight maps at each level.
static const int WEIGHT_BLUR_RADIUS = 5;

Task_PyramidMerge::Task_PyramidMerge(std::shared_ptr<Task_PyramidMerge> prev_merge,
                                     const std::vector<std::shared_ptr<ImgTask>>& images):
  m_prev(prev_merge), m_images(images)
{
  m_name = "PyramidMerge " + std::to_string(images.size()) + " images";
  m_filename = "pyramid_merge.jpg";

  if (prev_merge)
    m_depends_on.push_back(prev_merge);

  m_depends_on.insert(m_depends_on.end(), images.begin(), images.end());
}

int Task_PyramidMerge::levels_for_size(cv::Size size)
{
  // Reuse the same level count logic as the wavelet pipeline so pyramid
  // depth is consistent with image content.
  return Task_Wavelet::levels_for_size(size);
}

void Task_PyramidMerge::build_laplacian(const cv::Mat& img, std::vector<cv::Mat>& out_lap, int levels)
{
  out_lap.resize(levels);

  cv::Mat current;
  img.convertTo(current, CV_32F);

  for (int l = 0; l < levels - 1; l++)
  {
    cv::Mat down, up;
    cv::pyrDown(current, down);
    cv::pyrUp(down, up, current.size());
    out_lap[l] = current - up;
    current = down;
  }

  // Coarsest level: store the Gaussian remainder as-is.
  out_lap[levels - 1] = current.clone();
}

void Task_PyramidMerge::focus_weights(const std::vector<cv::Mat>& lap, std::vector<cv::Mat>& out_w)
{
  int levels = static_cast<int>(lap.size());
  out_w.resize(levels);

  for (int l = 0; l < levels; l++)
  {
    cv::Mat energy;
    if (lap[l].channels() == 1)
    {
      cv::multiply(lap[l], lap[l], energy);
    }
    else
    {
      // For color: sum squared channels to get scalar energy.
      std::vector<cv::Mat> ch;
      cv::split(lap[l], ch);
      energy = cv::Mat::zeros(lap[l].size(), CV_32F);
      for (auto& c : ch)
      {
        cv::Mat sq;
        cv::multiply(c, c, sq);
        energy += sq;
      }
    }

    // Smooth so that weight transitions are gradual (the "spline" part).
    int ksize = 2 * WEIGHT_BLUR_RADIUS + 1;
    cv::GaussianBlur(energy, out_w[l], cv::Size(ksize, ksize), 0);
  }
}

void Task_PyramidMerge::task()
{
  if (m_images.empty())
    throw std::runtime_error("Task_PyramidMerge: no input images");

  // Determine output size and level count from the first image's valid area.
  cv::Rect valid = m_images.front()->valid_area();
  // Intersection of all input valid areas.
  for (auto& img : m_images)
    valid &= img->valid_area();

  cv::Size full_size = m_images.front()->img().size();
  int levels = levels_for_size(full_size);

  // Initialise or clone accumulators from previous batch.
  if (m_prev)
  {
    m_num   = m_prev->m_num;
    m_denom = m_prev->m_denom;
    // Narrow valid area.
    valid &= m_prev->m_valid_area;
  }
  else
  {
    // First batch: allocate zero accumulators matching each pyramid level size.
    m_num.resize(levels);
    m_denom.resize(levels);

    cv::Size sz = full_size;
    int channels = m_images.front()->img().channels();
    for (int l = 0; l < levels; l++)
    {
      m_num[l]   = cv::Mat::zeros(sz, channels == 1 ? CV_32FC1 : CV_32FC3);
      m_denom[l] = cv::Mat::zeros(sz, CV_32FC1);
      sz = cv::Size((sz.width + 1) / 2, (sz.height + 1) / 2);
    }
  }

  m_valid_area = valid;

  // For each image in this batch, accumulate its contribution.
  for (auto& img_task : m_images)
  {
    const cv::Mat& img = img_task->img();

    std::vector<cv::Mat> lap;
    build_laplacian(img, lap, levels);

    std::vector<cv::Mat> weights;
    focus_weights(lap, weights);

    for (int l = 0; l < levels; l++)
    {
      if (lap[l].channels() == 1)
      {
        // Grayscale: simple multiply-accumulate.
        cv::Mat contrib;
        cv::multiply(lap[l], weights[l], contrib);
        m_num[l] += contrib;
      }
      else
      {
        // Color: broadcast scalar weight across channels.
        std::vector<cv::Mat> chans;
        cv::split(lap[l], chans);
        std::vector<cv::Mat> result_chans;
        for (auto& c : chans)
        {
          cv::Mat contrib;
          cv::multiply(c, weights[l], contrib);
          result_chans.push_back(contrib);
        }
        cv::Mat merged;
        cv::merge(result_chans, merged);
        m_num[l] += merged;
      }

      m_denom[l] += weights[l];
    }
  }

  // Release input references — no longer needed.
  m_images.clear();
  m_prev.reset();
}
