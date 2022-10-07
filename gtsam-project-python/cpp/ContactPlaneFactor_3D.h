#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class ContactPlaneFactor_3D: public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

public:

  ContactPlaneFactor_3D(gtsam::Key key1, gtsam::Key key2, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, key1, key2) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    gtsam::Point3 a = p1.rotation().column(2);
    gtsam::Point3 b = p2.translation() - p1.translation();
    
    gtsam::Matrix13 H;
    double rval = gtsam::dot(a, b, boost::none, (H1||H2) ? &H : 0);

    if (H1) *H1 = (gtsam::Matrix16() << 0,0,0,0,0,0).finished();
    H = p2.rotation().inverse().rotate(H);
    if (H2) *H2 = (gtsam::Matrix16() << 0, 0, 0, H(0,0), H(0,1), H(0,2)).finished();

    return (gtsam::Vector1() << rval).finished();
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class ContactPlaneFactor

} /// namespace gtsam_packing