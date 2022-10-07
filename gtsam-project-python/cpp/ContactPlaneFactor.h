#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class ContactPlaneFactor: public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2> {

private:

  double m_;

public:

  ContactPlaneFactor(gtsam::Key key1, gtsam::Key key2, const gtsam::Vector1 m, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2>(model, key1, key2), m_(m[0]) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Point2& p2,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    //if (H1) *H1 = (gtsam::Matrix13() << -std::cos(p1.theta()), -std::sin(p1.theta()), -std::sin(p1.theta())*(p2[0]-p1.x())+std::cos(p1.theta())*(p2[1]-p1.y())).finished();
    if (H1) *H1 = (gtsam::Matrix13() << -1, 0, -std::sin(p1.theta())*(p2[0]-p1.x())+std::cos(p1.theta())*(p2[1]-p1.y())).finished();
    if (H2) *H2 = (gtsam::Matrix12() << std::cos(p1.theta()), std::sin(p1.theta())).finished();

    return (gtsam::Vector1() << std::cos(p1.theta())*(p2[0]-p1.x())+std::sin(p1.theta())*(p2[1]-p1.y())-m_).finished();
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

}; // \class ContactPlaneFactor

} /// namespace gtsam_packing