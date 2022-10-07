#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

// Factor 1: previous object pose
// Factor 2: previous contact point
// Factor 3: current gripper pose
// Factor 4: current object pose
// Factor 5: current contact point
// Factor 6: [alpha, mu] (Vector2)
class InsideFrictionConeFactor: public gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Vector1> {

public:

  InsideFrictionConeFactor(gtsam::Key key1, gtsam::Key key2, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Vector1>(model, key1, key2){}

  gtsam::Vector evaluateError(const gtsam::Vector1& p1, const gtsam::Vector1& p2,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    // p1: cone angle arctan(\mu)
    // p2: offset angle theta

    if (std::abs(p2[0]) < p1[0]) {
      if (H1) *H1 = (gtsam::Matrix11() << 0).finished();
      if (H2) *H2 = (gtsam::Matrix11() << 0).finished();
      return (gtsam::Vector1() << 0).finished();
    } else if (p2[0] > p1[0]){
      if (H1) *H1 = (gtsam::Matrix11() << -1).finished();
      if (H2) *H2 = (gtsam::Matrix11() << 1).finished();
      return (gtsam::Vector1() << p2[0] - p1[0]).finished();
    } else {
      if (H1) *H1 = (gtsam::Matrix11() << -1).finished();
      if (H2) *H2 = (gtsam::Matrix11() << -1).finished();
      return (gtsam::Vector1() << - p1[0] - p2[0]).finished();
    }

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class TactileTransformFactor

} /// namespace gtsam_packing