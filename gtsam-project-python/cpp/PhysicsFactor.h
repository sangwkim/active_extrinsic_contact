#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class PhysicsFactor: public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Vector1> {

private:

  double mx_, my_, mtheta_;

public:

  PhysicsFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    const gtsam::Pose2 m, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Vector1>(model, key1, key2, key3, key4),
      mx_(m.x()), my_(m.y()), mtheta_(m.theta()) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Pose2& p2,
    const gtsam::Point2& p3, const gtsam::Vector1& p4, boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none,
    boost::optional<gtsam::Matrix&> H4 = boost::none) const {

    gtsam::Pose2 p1_2 = gtsam::traits<gtsam::Pose2>::Between(p2,p1,H2,H1);
    double r = - p1_2.y() - std::sin(p2.theta())*(p3[0]-p2.x()) + std::cos(p2.theta())*(p3[1]-p2.y());
    double tr_rot = std::pow(std::pow(mx_,2)+std::pow(my_,2),0.5) / mtheta_;

    gtsam::Matrix Hr12 = (gtsam::Matrix13() << 0, -1, 0).finished();
    gtsam::Matrix Hr2 = (gtsam::Matrix13() << std::sin(p2.theta()), -std::cos(p2.theta()), -std::cos(p2.theta())*(p3[0]-p2.x()) - std::sin(p2.theta())*(p3[1]-p2.y())).finished();
    //typename gtsam::traits<gtsam::Pose2>::ChartJacobian::Jacobian Hr12, Hr2;
    //Hr12 = (gtsam::Matrix13() << 0, -1, 0).finished();
    //Hr2 = (gtsam::Matrix13() << std::sin(p2.theta()), -std::cos(p2.theta()), -std::cos(p2.theta())*(p3.x()-p2.x()) - std::sin(p2.theta())*(p3.y()-p2.y())).finished();
    if (H1) *H1 = tr_rot * Hr12 * (*H1);
    if (H2) *H2 = tr_rot * ((Hr12 * (*H2)) + Hr2);
    if (H3) *H3 = tr_rot * (gtsam::Matrix12() << -std::sin(p2.theta()), std::cos(p2.theta())).finished();
    if (H4) *H4 = (gtsam::Matrix11() << -1).finished();
    
    
    std::cout << "-------------" << std::endl;
    std::cout << "gripper: " << p1 << std::endl;
    std::cout << "object: " << p2 << std::endl;
    std::cout << "contact: " << p3 << std::endl;
    std::cout << "alpha: " << p4 << std::endl;
    std::cout << "moment arm r: " << r << std::endl;
    std::cout << "trans/rot: " << tr_rot << std::endl;
    std::cout << "est alpha: " << r * tr_rot << std::endl;
    if (H1) std::cout << "H_gripper: " << *H1 << std::endl;
    if (H2) std::cout << "H_object: " << *H2 << std::endl;
    //std::cout << (gtsam::Vector1() << r * trns - p4[0] * mtheta_).finished() << std::endl;
    

    return (gtsam::Vector1() << r * tr_rot - p4[0]).finished();

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class TactileTransformFactor

} /// namespace gtsam_packing