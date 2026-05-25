// Collapses a Burt-Adelson Laplacian pyramid accumulated by Task_PyramidMerge
// into a final blended color image.
//
// The accumulated num/denom pyramids are normalized per-level, then reconstructed
// from coarsest to finest using pyrUp + add.

#pragma once
#include "worker.hh"
#include "task_pyramidmerge.hh"

namespace focusstack {

class Task_PyramidCollapse: public ImgTask
{
public:
  explicit Task_PyramidCollapse(std::shared_ptr<Task_PyramidMerge> merged);

private:
  virtual void task();

  std::shared_ptr<Task_PyramidMerge> m_merged;
};

}
