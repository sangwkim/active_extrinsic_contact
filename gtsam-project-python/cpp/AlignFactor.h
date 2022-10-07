#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class AlignFactor: public gtsam::NoiseModelFactor1<gtsam::Pose3> {

public:

  AlignFactor(gtsam::Key key1, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key1) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, boost::optional<gtsam::Matrix&> H1 = boost::none) const {

    const double norm = std::pow(std::pow(p1.x(),2)+std::pow(p1.y(),2)+std::pow(p1.z(),2),0.5);

    //typename gtsam::traits<gtsam::Point3>::ChartJacobian::Jacobian H;
    gtsam::Matrix33 H;
    gtsam::Matrix13 Hx, Hy, Hz;
    gtsam::Vector rval = gtsam::traits<gtsam::Point3>::Between(p1.rotation().column(2) * norm,
             p1.translation(), boost::none, H1 ? &H : 0);

    Hx = p1.rotation().inverse().rotate((gtsam::Matrix13() << H(0,0), H(0,1), H(0,2)).finished());
    Hy = p1.rotation().inverse().rotate((gtsam::Matrix13() << H(1,0), H(1,1), H(1,2)).finished());
    Hz = p1.rotation().inverse().rotate((gtsam::Matrix13() << H(2,0), H(2,1), H(2,2)).finished());

    if (H1) *H1 = (gtsam::Matrix36() << 0.0, 0.0, 0.0, Hx(0,0), Hx(0,1), Hx(0,2),
                                        0.0, 0.0, 0.0, Hy(0,0), Hy(0,1), Hy(0,2),
                                        0.0, 0.0, 0.0, Hz(0,0), Hz(0,1), Hz(0,2)).finished();
                                      //Rx,  Ry,  Rz,  Tx,      Ty,      Tz

    return rval;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 1;
  }

}; // 

} /// namespace gtsam_packing