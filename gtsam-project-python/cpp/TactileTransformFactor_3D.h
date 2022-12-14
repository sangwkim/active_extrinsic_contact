#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class TactileTransformFactor_3D: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

private:

  gtsam::Pose3 measured_;

public:

  TactileTransformFactor_3D(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    const gtsam::Pose3 m, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3, key4),
      measured_(m) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2,
    const gtsam::Pose3& p3, const gtsam::Pose3& p4, boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none,
    boost::optional<gtsam::Matrix&> H4 = boost::none) const {
      
      gtsam::Pose3 p3_1 = gtsam::traits<gtsam::Pose3>::Between(p1,p3,H1,H3);
      gtsam::Pose3 p4_2 = gtsam::traits<gtsam::Pose3>::Between(p2,p4,H2,H4);
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H13;
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian H24;
      gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(p3_1,p4_2,&H13,&H24);
      if (H1) *H1 = H13 * (*H1);
      if (H2) *H2 = H24 * (*H2);
      if (H3) *H3 = H13 * (*H3);
      if (H4) *H4 = H24 * (*H4);
#ifdef SLOW_BUT_CORRECT_BETWEENFACTOR
      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hlocal;
      gtsam::Vector rval = gtsam::traits<gtsam::Pose3>::Local(measured_, hx, boost::none, (H1 || H2 || H3 || H4) ? &Hlocal : 0);
      if (H1) *H1 = Hlocal * (*H1);
      if (H2) *H2 = Hlocal * (*H2);
      if (H3) *H3 = Hlocal * (*H3);
      if (H4) *H4 = Hlocal * (*H4);
      return rval
#else
      return gtsam::traits<gtsam::Pose3>::Local(measured_, hx);
#endif
  }

  /** return the measured */
  const gtsam::Pose3& measured() const {
    return measured_;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class TactileTransformFactor_3D

} /// namespace gtsam_packing