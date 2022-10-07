#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class ClineCenterFactor: public gtsam::NoiseModelFactor1<gtsam::Pose3> {

public:

  ClineCenterFactor(gtsam::Key key1, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key1) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1,
    boost::optional<gtsam::Matrix&> H1 = boost::none) const {

    gtsam::Point3 a = p1.translation();
    gtsam::Point3 b = p1.rotation().column(1);

    gtsam::Matrix13 H;
    double rval = gtsam::dot(a, b, H1 ? &H : 0, boost::none);
    H = p1.rotation().inverse().rotate(H);

    if (H1) *H1 = (gtsam::Matrix16() << 0, 0, 0, H(0,0), H(0,1), H(0,2)).finished();

    return (gtsam::Vector1() << rval).finished();
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 1;
  }

}; // \class 

} /// namespace gtsam_packing