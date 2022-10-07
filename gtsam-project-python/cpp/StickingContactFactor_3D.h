#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class StickingContactFactor_3D: public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {

public:

  StickingContactFactor_3D(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, gtsam::Key key4,
    gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, key1, key2, key3, key4) {}

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

      typename gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian HLM;
      gtsam::Vector6 lm = gtsam::traits<gtsam::Pose3>::Logmap(hx, &HLM);
      if (H1) *H1 = HLM * (*H1);
      if (H2) *H2 = HLM * (*H2);
      if (H3) *H3 = HLM * (*H3);
      if (H4) *H4 = HLM * (*H4);

      return lm;

  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 4;
  }

}; // \class StickingContactFactor_3D

} /// namespace gtsam_packing