#pragma once

#include <ostream>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Lie.h>

namespace gtsam_packing {

class ContactLineParallelFactor: public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {

public:

  ContactLineParallelFactor(gtsam::Key key1, gtsam::Key key2, gtsam::SharedNoiseModel model) :
      gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, key1, key2) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& p1, const gtsam::Pose3& p2,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    gtsam::Matrix66 H;
    gtsam::Pose3 p_btw = gtsam::traits<gtsam::Pose3>::Between(p1,p2, boost::none, (H1||H2) ? &H : 0);

    //if (H1) *H1 = (gtsam::Matrix26() << *H1(2,0), *H1(2,1), *H1(2,2), *H1(2,3), *H1(2,4), *H1(2,5), 
    //                                    *H1(4,0), *H1(4,1), *H1(4,2), *H1(4,3), *H1(4,4), *H1(4,5)).finished();
    if (H1) *H1 = (gtsam::Matrix26() << 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0).finished();
    std::cout << p_btw << std::endl << p_btw.rotation().ypr() << std::endl << p_btw.rotation().yaw() << std::endl;
    if (H2) *H2 = (gtsam::Matrix26() << 0, 0, 1, 0, 0, 0, 
                                        0, 0, 0, 0, 1, 0).finished();

    return (gtsam::Vector2() << p_btw.rotation().yaw(), p_btw.y()).finished();
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

};

} /// namespace gtsam_packing