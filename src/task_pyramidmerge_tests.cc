#include <gtest/gtest.h>
#include "task_pyramidmerge.hh"
#include "task_pyramidcollapse.hh"
#include "task_wavelet.hh"
#include "logger.hh"
#include <opencv2/imgproc.hpp>

namespace focusstack {

// Helper: create an ImgTask that wraps a pre-built cv::Mat.
static std::shared_ptr<ImgTask> make_img_task(const cv::Mat& img)
{
  auto t = std::make_shared<ImgTask>(img);
  t->run();  // mark as completed so it satisfies dependency checks
  return t;
}

// Laplacian pyramid decompose then collapse should reconstruct the original.
TEST(Task_PyramidMerge, RoundtripGrayscale)
{
  cv::Mat input(64, 64, CV_8UC1);
  cv::randu(input, 0, 256);

  int levels = Task_PyramidMerge::levels_for_size(input.size());

  std::vector<cv::Mat> lap;
  Task_PyramidMerge::build_laplacian(input, lap, levels);
  ASSERT_EQ((int)lap.size(), levels);

  // Collapse manually.
  cv::Mat result = lap[levels - 1].clone();
  for (int l = levels - 2; l >= 0; l--)
  {
    cv::Mat up;
    cv::pyrUp(result, up, lap[l].size());
    result = up + lap[l];
  }

  // Convert back to uint8 for comparison.
  cv::Mat reconstructed;
  result.convertTo(reconstructed, CV_8U);

  // Reconstruction error should be small (< 2 counts per pixel on average).
  cv::Mat diff;
  cv::absdiff(input, reconstructed, diff);
  double mean_err = cv::mean(diff)[0];
  EXPECT_LT(mean_err, 2.0);
}

TEST(Task_PyramidMerge, RoundtripColor)
{
  cv::Mat input(64, 64, CV_8UC3);
  cv::randu(input, 0, 256);

  int levels = Task_PyramidMerge::levels_for_size(input.size());

  std::vector<cv::Mat> lap;
  Task_PyramidMerge::build_laplacian(input, lap, levels);
  ASSERT_EQ((int)lap.size(), levels);

  cv::Mat result = lap[levels - 1].clone();
  for (int l = levels - 2; l >= 0; l--)
  {
    cv::Mat up;
    cv::pyrUp(result, up, lap[l].size());
    result = up + lap[l];
  }

  cv::Mat reconstructed;
  result.convertTo(reconstructed, CV_8U);

  cv::Mat diff;
  cv::absdiff(input, reconstructed, diff);
  double mean_err = cv::mean(diff)[0];
  EXPECT_LT(mean_err, 2.0);
}

// Merging a single image and collapsing should reproduce that image.
TEST(Task_PyramidMerge, SingleImageIdentity)
{
  cv::Mat input(64, 64, CV_8UC3);
  cv::randu(input, 0, 256);

  std::shared_ptr<Logger> logger = std::make_shared<Logger>();

  auto img_task = make_img_task(input);

  auto merge = std::make_shared<Task_PyramidMerge>(nullptr, std::vector<std::shared_ptr<ImgTask>>{img_task});
  merge->run(logger);

  auto collapse = std::make_shared<Task_PyramidCollapse>(merge);
  collapse->run(logger);

  const cv::Mat& result = collapse->img();
  ASSERT_EQ(result.size(), input.size());
  ASSERT_EQ(result.type(), CV_8UC3);

  // Single-image blend should reproduce the input closely.
  cv::Mat diff;
  cv::absdiff(input, result, diff);
  double mean_err = cv::mean(diff)[0];
  EXPECT_LT(mean_err, 2.0);
}

// Merging two identical images should give the same result as one.
TEST(Task_PyramidMerge, TwoIdenticalImages)
{
  cv::Mat input(64, 64, CV_8UC3);
  cv::randu(input, 0, 256);

  std::shared_ptr<Logger> logger = std::make_shared<Logger>();

  auto t1 = make_img_task(input);
  auto t2 = make_img_task(input);

  auto merge = std::make_shared<Task_PyramidMerge>(nullptr, std::vector<std::shared_ptr<ImgTask>>{t1, t2});
  merge->run(logger);

  auto collapse = std::make_shared<Task_PyramidCollapse>(merge);
  collapse->run(logger);

  const cv::Mat& result = collapse->img();

  cv::Mat diff;
  cv::absdiff(input, result, diff);
  double mean_err = cv::mean(diff)[0];
  EXPECT_LT(mean_err, 2.0);
}

// When one image is sharp and one is blurred, the merge should be closer to the sharp image.
TEST(Task_PyramidMerge, PrefersSharpImage)
{
  cv::Mat sharp(64, 64, CV_8UC3);
  sharp = cv::Scalar(128, 128, 128);
  // Draw a sharp edge.
  sharp(cv::Rect(32, 0, 32, 64)) = cv::Scalar(200, 200, 200);

  cv::Mat blurred;
  cv::GaussianBlur(sharp, blurred, cv::Size(15, 15), 5.0);

  std::shared_ptr<Logger> logger = std::make_shared<Logger>();

  auto t_sharp  = make_img_task(sharp);
  auto t_blurred = make_img_task(blurred);

  auto merge = std::make_shared<Task_PyramidMerge>(nullptr,
    std::vector<std::shared_ptr<ImgTask>>{t_sharp, t_blurred});
  merge->run(logger);

  auto collapse = std::make_shared<Task_PyramidCollapse>(merge);
  collapse->run(logger);

  const cv::Mat& result = collapse->img();

  // Result should be closer to the sharp image than to the blurred one.
  cv::Mat diff_sharp, diff_blurred;
  cv::absdiff(result, sharp,   diff_sharp);
  cv::absdiff(result, blurred, diff_blurred);

  double err_vs_sharp   = cv::mean(diff_sharp)[0];
  double err_vs_blurred = cv::mean(diff_blurred)[0];

  EXPECT_LT(err_vs_sharp, err_vs_blurred);
}

TEST(Task_PyramidMerge, LevelsForSize)
{
  // Small image: minimum levels
  int l_small = Task_PyramidMerge::levels_for_size(cv::Size(32, 32));
  int min_levels = Task_Wavelet::min_levels;
  EXPECT_GE(l_small, min_levels);

  // Large image: more levels but capped at max
  int l_large = Task_PyramidMerge::levels_for_size(cv::Size(4096, 4096));
  int max_levels = Task_Wavelet::max_levels;
  EXPECT_LE(l_large, max_levels);
  EXPECT_GE(l_large, l_small);
}

}
