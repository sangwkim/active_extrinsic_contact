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
class FrictionConeFactor: public gtsam::NoiseModelFactor6<gtsam::Pose2, gtsam::Point2,
                                      gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Vector2> {

private:

  double mx_, my_, mtheta_;

public:

  FrictionConeFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
                  gtsam::Key key5, gtsam::Key key6, const gtsam::Pose2 m, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor6<gtsam::Pose2, gtsam::Point2, gtsam::Pose2, gtsam::Pose2,
                        gtsam::Point2, gtsam::Vector2>(model, key1, key2, key3, key4, key5, key6),
      mx_(m.x()), my_(m.y()), mtheta_(m.theta()) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Point2& p2,
    const gtsam::Pose2& p3, const gtsam::Pose2& p4, const gtsam::Point2& p5, const gtsam::Vector2& p6,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none,
    boost::optional<gtsam::Matrix&> H3 = boost::none, boost::optional<gtsam::Matrix&> H4 = boost::none,
    boost::optional<gtsam::Matrix&> H5 = boost::none, boost::optional<gtsam::Matrix&> H6 = boost::none) const {

    // slip_dir 
    // True: slipping up
    // False: slipping down
    bool slip_dir = (std::cos(p1.theta())*(p2[1]-p1.y()) - std::sin(p1.theta())*(p2[0]-p1.x()))
                     > (std::cos(p4.theta())*(p5[1]-p4.y()) - std::sin(p4.theta())*(p5[0]-p4.x()));

    double cone_angle =  std::atan(p6[1]);

    if (~slip_dir) cone_angle = (-1) * cone_angle;

    double r = std::cos(p4.theta()+cone_angle)*(p5[1]-p3.y()) - std::sin(p4.theta()+cone_angle)*(p5[0]-p3.x());

    gtsam::Matrix Hr3 = (gtsam::Matrix13() << std::sin(p4.theta()+cone_angle), -std::cos(p4.theta()+cone_angle), 0).finished();
    gtsam::Matrix Hr4 = (gtsam::Matrix13() << 0, 0, -std::sin(p4.theta()+cone_angle)*(p5[1]-p3.y()) - std::cos(p4.theta()+cone_angle)*(p5[0]-p3.x())).finished();
    
    if (H1) *H1 = (gtsam::Matrix23() << 0,0,0,0,0,0).finished();
    if (H2) *H2 = (gtsam::Matrix22() << 0,0,0,0).finished();
    if (H3) *H3 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * Hr3
                    + (gtsam::Matrix23() << 0, 0,  p6[0]*std::sin(p4.theta()+cone_angle-p3.theta()),
                                            0, 0, -p6[0]*std::cos(p4.theta()+cone_angle-p3.theta())).finished();
    if (H4) *H4 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * Hr4
                    + (gtsam::Matrix23() << 0, 0, -p6[0]*std::sin(p4.theta()+cone_angle-p3.theta()),
                                            0, 0,  p6[0]*std::cos(p4.theta()+cone_angle-p3.theta())).finished();
    if (H5) *H5 = (gtsam::Matrix21() << mx_/mtheta_, my_/mtheta_).finished() * (gtsam::Matrix12() << -std::sin(p4.theta()+cone_angle), std::cos(p4.theta()+cone_angle)).finished();
    if (H6) *H6 = (gtsam::Matrix21() << std::cos(p4.theta()+cone_angle-p3.theta()), std::sin(p4.theta()+cone_angle-p3.theta())).finished();

    return (gtsam::Vector2() << (p6[0] * std::cos(p4.theta()+cone_angle-p3.theta())) + (mx_ / mtheta_ * r),
                                (p6[0] * std::sin(p4.theta()+cone_angle-p3.theta())) + (my_ / mtheta_ * r)).finished();

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 6;
  }

}; // \class TactileTransformFactor

} /// namespace gtsam_packing