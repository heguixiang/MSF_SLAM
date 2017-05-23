/*
 * Copyright (C) 2012-2013 Simon Lynen, ASL, ETH Zurich, Switzerland
 * You can contact the author at <slynen at ethz dot ch>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MSF_TOOLS_H_
#define MSF_TOOLS_H_

#include <Eigen/Dense>
#include <algorithm>

namespace msf_core {
/***
 * Computes the median of a given vector.
 */
template<typename D>
typename Eigen::MatrixBase<D>::Scalar GetMedian(
    const Eigen::MatrixBase<D> & data) {
  static_assert(
      (Eigen::MatrixBase<D>::ColsAtCompileTime == 1),
      "GetMedian only takes Eigen column vectors as arguments");
  Eigen::Matrix<typename Eigen::MatrixBase<D>::Scalar,
      Eigen::MatrixBase<D>::RowsAtCompileTime,
      // Copy so we don't sort the original vector.
      Eigen::MatrixBase<D>::ColsAtCompileTime> m = data;

  if (Eigen::MatrixBase<D>::SizeAtCompileTime) {
    double * begin = m.data();
    double * end = m.data() + m.SizeAtCompileTime;
    double * middle = begin + static_cast<int>(std::floor((end - begin) / 2));
    std::nth_element(begin, middle, end);
    return *middle;
  } else
    return 0;
}

/***
 * Outputs the time in seconds in a human readable format for debugging.
 */
double timehuman(double val);

}

#endif  // MSF_TOOLS_H_
