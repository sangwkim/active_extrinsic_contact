#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class FixedRotationFactor: public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Point2> {

private:

  gtsam::Pose2 measured_1;
  gtsam::Pose2 measured_2;

public:

  FixedRotationFactor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    const gtsam::Pose2 m_1, const gtsam::Pose2 m_2, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Point2, gtsam::Point2>(model, key1, key2, key3, key4),
      measured_1(m_1), measured_2(m_2) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Pose2& p2,
    const gtsam::Point2& p3, const gtsam::Point2& p4, boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none,
    boost::optional<gtsam::Matrix&> H4 = boost::none) const {
      
      gtsam::Pose2 p1_ = gtsam::traits<gtsam::Pose2>::Compose(p1,measured_1,H1,boost::none);
      gtsam::Pose2 p2_ = gtsam::traits<gtsam::Pose2>::Compose(p2,measured_2,H2,boost::none);
      
      /*
      if (H1) *H1 = (gtsam::Matrix23() << -std::cos(p1_.theta()), -std::sin(p1_.theta()), -std::sin(p1_.theta())*(p3[0]-p1_.x())+std::cos(p1_.theta())*(p3[1]-p1_.y()),
                                           std::sin(p1_.theta()), -std::cos(p1_.theta()), -std::cos(p1_.theta())*(p3[0]-p1_.x())-std::sin(p1_.theta())*(p3[1]-p1_.y())).finished()
                                          * (*H1);
      if (H2) *H2 = (gtsam::Matrix23() << std::cos(p2_.theta()), std::sin(p2_.theta()), std::sin(p2_.theta())*(p4[0]-p2_.x())-std::cos(p2_.theta())*(p4[1]-p2_.y()),
                                         -std::sin(p2_.theta()), std::cos(p2_.theta()), std::cos(p2_.theta())*(p4[0]-p2_.x())+std::sin(p2_.theta())*(p4[1]-p2_.y())).finished()
                                          * (*H2);
      */
      if (H1) *H1 = (gtsam::Matrix23() << -1, 0, -std::sin(p1_.theta())*(p3[0]-p1_.x())+std::cos(p1_.theta())*(p3[1]-p1_.y()),
                                           0, -1, -std::cos(p1_.theta())*(p3[0]-p1_.x())-std::sin(p1_.theta())*(p3[1]-p1_.y())).finished()
                                          * (*H1);
      if (H2) *H2 = (gtsam::Matrix23() << 1, 0, std::sin(p2_.theta())*(p4[0]-p2_.x())-std::cos(p2_.theta())*(p4[1]-p2_.y()),
                                         0, 1, std::cos(p2_.theta())*(p4[0]-p2_.x())+std::sin(p2_.theta())*(p4[1]-p2_.y())).finished()
                                          * (*H2);
      if (H3) *H3 = (gtsam::Matrix22() << std::cos(p1_.theta()), std::sin(p1_.theta()),
                                         -std::sin(p1_.theta()), std::cos(p1_.theta())).finished();
      if (H4) *H4 = (gtsam::Matrix22() << -std::cos(p2_.theta()), -std::sin(p2_.theta()),
                                          std::sin(p2_.theta()), -std::cos(p2_.theta())).finished();

      return (gtsam::Vector2() << std::cos(p1_.theta())*(p3[0]-p1_.x())+std::sin(p1_.theta())*(p3[1]-p1_.y())
                                -std::cos(p2_.theta())*(p4[0]-p2_.x())-std::sin(p2_.theta())*(p4[1]-p2_.y()),
                                -std::sin(p1_.theta())*(p3[0]-p1_.x())+std::cos(p1_.theta())*(p3[1]-p1_.y())
                                +std::sin(p2_.theta())*(p4[0]-p2_.x())-std::cos(p2_.theta())*(p4[1]-p2_.y())).finished();

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class FixedRotationFactor

} /// namespace gtsam_packing