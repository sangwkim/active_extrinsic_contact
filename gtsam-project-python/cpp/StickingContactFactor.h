#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class StickingContactFactor: public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Point2> {

public:

  StickingContactFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Point2>(model, key1, key2, key3, key4){}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Pose2& p2,
    const gtsam::Point2& p3, const gtsam::Point2& p4, boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none,
    boost::optional<gtsam::Matrix&> H4 = boost::none) const {
      
    if (H1) *H1 = (gtsam::Matrix23() << -1, 0, -std::sin(p1.theta())*(p3[0]-p1.x())+std::cos(p1.theta())*(p3[1]-p1.y()),
                                         0, -1, -std::cos(p1.theta())*(p3[0]-p1.x())-std::sin(p1.theta())*(p3[1]-p1.y())).finished();
    if (H2) *H2 = (gtsam::Matrix23() << 1, 0, std::sin(p2.theta())*(p4[0]-p2.x())-std::cos(p2.theta())*(p4[1]-p2.y()),
                                       0, 1, std::cos(p2.theta())*(p4[0]-p2.x())+std::sin(p2.theta())*(p4[1]-p2.y())).finished();
    if (H3) *H3 = (gtsam::Matrix22() << std::cos(p1.theta()), std::sin(p1.theta()),
                                       -std::sin(p1.theta()), std::cos(p1.theta())).finished();
    if (H4) *H4 = (gtsam::Matrix22() << -std::cos(p2.theta()), -std::sin(p2.theta()),
                                        std::sin(p2.theta()), -std::cos(p2.theta())).finished();

    return (gtsam::Vector2() << std::cos(p1.theta())*(p3[0]-p1.x())+std::sin(p1.theta())*(p3[1]-p1.y())
                                -std::cos(p2.theta())*(p4[0]-p2.x())-std::sin(p2.theta())*(p4[1]-p2.y()),
                                -std::sin(p1.theta())*(p3[0]-p1.x())+std::cos(p1.theta())*(p3[1]-p1.y())
                                +std::sin(p2.theta())*(p4[0]-p2.x())-std::cos(p2.theta())*(p4[1]-p2.y())).finished();

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class StickingContactFactor

} /// namespace gtsam_packing