#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class PhysicsFactor_v2: public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Vector1> {

private:

  double mx_, my_, mtheta_;

public:

  PhysicsFactor_v2(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    const gtsam::Pose2 m, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Vector1>(model, key1, key2, key3, key4),
      mx_(m.x()), my_(m.y()), mtheta_(m.theta()) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Pose2& p2,
    const gtsam::Point2& p3, const gtsam::Vector1& p4, boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none,
    boost::optional<gtsam::Matrix&> H4 = boost::none) const {

    //gtsam::Pose2 p1_2 = gtsam::traits<gtsam::Pose2>::Between(p2,p1,H2,H1);
    //double r = - p1_2.y() - std::sin(p2.theta())*(p3[0]-p2.x()) + std::cos(p2.theta())*(p3[1]-p2.y());
    double r = std::cos(p2.theta())*(p3[1]-p1.y()) - std::sin(p2.theta())*(p3[0]-p1.x());

    gtsam::Matrix Hr1 = (gtsam::Matrix13() << std::sin(p2.theta()), -std::cos(p2.theta()), 0).finished();
    gtsam::Matrix Hr2 = (gtsam::Matrix13() << 0, 0, -std::sin(p2.theta())*(p3[1]-p1.y()) - std::cos(p2.theta())*(p3[0]-p1.x())).finished();
    
    if (H1) *H1 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * Hr1
                    + (gtsam::Matrix23() << 0, 0,  p4[0]*std::sin(p2.theta()-p1.theta()),
                                            0, 0, -p4[0]*std::cos(p2.theta()-p1.theta())).finished();
    if (H2) *H2 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * Hr2
                    + (gtsam::Matrix23() << 0, 0, -p4[0]*std::sin(p2.theta()-p1.theta()),
                                            0, 0,  p4[0]*std::cos(p2.theta()-p1.theta())).finished();
    if (H3) *H3 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * (gtsam::Matrix12() << -std::sin(p2.theta()), std::cos(p2.theta())).finished();
    if (H4) *H4 = (gtsam::Matrix21() << std::cos(p2.theta()-p1.theta()), std::sin(p2.theta()-p1.theta())).finished();

    return (gtsam::Vector2() << (p4[0] * std::cos(p2.theta()-p1.theta())) + (mx_ / mtheta_ * r),
                                (p4[0] * std::sin(p2.theta()-p1.theta())) + (my_ / mtheta_ * r)).finished();

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class TactileTransformFactor

} /// namespace gtsam_packing