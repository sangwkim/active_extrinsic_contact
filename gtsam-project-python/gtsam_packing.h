/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * 
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file     gtsam_example.h
 * @brief    Example wrapper interface file for Python
 * @author   Varun Agrawal
 */

// This is an interface file for automatic Python wrapper generation.
// See gtsam.h for full documentation and more examples.

#include <cpp/greeting.h>
#include <cpp/GPSPose2Factor.h>
#include <cpp/StickingContactFactor.h>
#include <cpp/StickingContactFactor_3D.h>
#include <cpp/TactileTransformFactor.h>
#include <cpp/TactileTransformFactor_3D.h>
#include <cpp/ContactPlaneFactor.h>
#include <cpp/ContactPlaneFactor_3D.h>
#include <cpp/PhysicsFactor.h>
#include <cpp/PhysicsFactor_v2.h>
#include <cpp/PhysicsFactor_v2_2.h>
#include <cpp/PhysicsFactor_v3.h>
#include <cpp/FrictionConeFactor.h>
#include <cpp/InsideFrictionConeFactor.h>
#include <cpp/AlignFactor.h>
#include <cpp/ContactLineParallelFactor.h>
#include <cpp/FixedRotationFactor.h>
#include <cpp/ClineCenterFactor.h>

// The namespace should be the same as in the c++ source code.
namespace gtsam_packing {

virtual class ClineCenterFactor : gtsam::NoiseModelFactor {
  ClineCenterFactor(size_t key1,
    const gtsam::noiseModel::Base* model);
};

virtual class FixedRotationFactor : gtsam::NoiseModelFactor {
  FixedRotationFactor(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose2& m_1, const gtsam::Pose2& m_2,
    const gtsam::noiseModel::Base* model);
};

virtual class StickingContactFactor : gtsam::NoiseModelFactor {
  StickingContactFactor(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model);
};

virtual class StickingContactFactor_3D : gtsam::NoiseModelFactor {
  StickingContactFactor_3D(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::noiseModel::Base* model);
};

virtual class TactileTransformFactor : gtsam::NoiseModelFactor {
  TactileTransformFactor(size_t key1, size_t key2, size_t key3, size_t key4,
  	const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class TactileTransformFactor_3D : gtsam::NoiseModelFactor {
  TactileTransformFactor_3D(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose3& m, const gtsam::noiseModel::Base* model);
};

virtual class ContactPlaneFactor : gtsam::NoiseModelFactor {
  ContactPlaneFactor(size_t key1, size_t key2, const gtsam::Vector1& m, const gtsam::noiseModel::Base* model);
};

virtual class ContactPlaneFactor_3D : gtsam::NoiseModelFactor {
  ContactPlaneFactor_3D(size_t key1, size_t key2, const gtsam::noiseModel::Base* model);
};

virtual class PhysicsFactor : gtsam::NoiseModelFactor {
  PhysicsFactor(size_t key1, size_t key2, size_t key3, size_t key4,
  	const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class PhysicsFactor_v2 : gtsam::NoiseModelFactor {
  PhysicsFactor_v2(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class PhysicsFactor_v2_2 : gtsam::NoiseModelFactor {
  PhysicsFactor_v2_2(size_t key1, size_t key2, size_t key3, size_t key4,
    const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class PhysicsFactor_v3 : gtsam::NoiseModelFactor {
  PhysicsFactor_v3(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5,
    const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class FrictionConeFactor : gtsam::NoiseModelFactor {
  FrictionConeFactor(size_t key1, size_t key2, size_t key3, size_t key4, size_t key5, size_t key6,
    const gtsam::Pose2& m, const gtsam::noiseModel::Base* model);
};

virtual class InsideFrictionConeFactor : gtsam::NoiseModelFactor {
  InsideFrictionConeFactor(size_t key1, size_t key2,
    const gtsam::noiseModel::Base* model);
};

virtual class GPSPose2Factor : gtsam::NoiseModelFactor {
  GPSPose2Factor(size_t poseKey, const gtsam::Point2& m, gtsam::noiseModel::Base* model);
};

virtual class AlignFactor : gtsam::NoiseModelFactor {
  AlignFactor(size_t key1, const gtsam::noiseModel::Base* model);
};

virtual class ContactLineParallelFactor : gtsam::NoiseModelFactor {
  ContactLineParallelFactor(size_t key1, size_t key2, const gtsam::noiseModel::Base* model);
};

class Greeting {
  Greeting();
  void sayHello() const;
  gtsam::Rot3 invertRot3(gtsam::Rot3 rot) const;
  void sayGoodbye() const;
};

}  // namespace gtsam_example
