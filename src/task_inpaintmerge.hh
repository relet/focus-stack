// Fills "unfocused" holes in a wavelet-domain merged image.
//
// Task_Merge selects the wavelet coefficient with the highest absolute value
// across all input frames, but never-focused pixels still get an arbitrary
// coefficient (from whichever frame happened to win).  Task_InpaintMerge
// identifies pixels whose winning coefficient is below a user-specified
// threshold and fills them by Gaussian-weighted averaging from neighbouring
// confident pixels — identical in spirit to the depth-map inpainting in
// Task_Depthmap_Inpaint.
//
// The result is a wavelet image with holes filled and drops directly into the
// existing inverse-wavelet → reassign pipeline.

#pragma once
#include "worker.hh"
#include "task_merge.hh"

namespace focusstack {

class Task_InpaintMerge: public ImgTask
{
public:
  // threshold: minimum wavelet-coefficient magnitude required for a pixel to
  // be considered "in focus".  Pixels below this are filled from neighbours.
  // A value of 0 (default) disables inpainting entirely.
  Task_InpaintMerge(std::shared_ptr<Task_Merge> merge, float threshold);

private:
  virtual void task();

  // Fill hole pixels in a single-channel float image using a Gaussian-weighted
  // average of the nearest valid (non-hole) neighbours.
  // valid_mask: CV_8UC1, 255 = confident pixel, 0 = hole to fill.
  // The result is written back to output; confident pixels are never modified.
  static void fill_holes(const cv::Mat& input, cv::Mat& output,
                         const cv::Mat& valid_mask, int radius);

  std::shared_ptr<Task_Merge> m_merge;
  float m_threshold;
};

}
