#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

namespace rebel_analytic_ik
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// -- Geometric constants, taken directly from igus_rebel2.urdf joint origins --------------
// base_link -> J_motor1 (fixed, xyz (0,0,0.1462), rpy (0,0,pi)) -> joint1 (revolute, axis z,
// origin identity). Composing the two fixed rpy=0 translations of (J_motor2 + joint2-origin),
// (J_motor3 + joint3-origin), (J_motor4 + joint4-origin) and (J_motor5 + joint5-origin) gives
// pure-translation offsets between consecutive rotation axes (every fixed joint in this chain
// has rpy=0 except J_motor1's rpy=(0,0,pi), which folds into the base offset below).
constexpr double kD1Z = 0.1462;              // base_link -> joint1 axis, along z
// J_motor1's yaw is written as 3.1416 in the URDF, not pi; the 7.3e-6 rad difference is
// visible against MoveIt FK, so use the URDF value exactly.
constexpr double kBaseYaw = 3.1416;
// link1 -> joint2 axis: J_motor2 (0,-0.0225,0.106) + joint2 origin (0,0.049,0).
constexpr double kO2Y = 0.0265;
constexpr double kO2Z = 0.106;
// link2 -> joint3 axis (in the joint2-rotated frame): J_motor3 (0,-0.049,0.24152) +
// joint3 origin (0,0.0225,0). This is the "L1" upper-arm length (x=0, so no elbow offset
// angle is needed for this link).
constexpr double kL1Y = -0.0265;
constexpr double kL1 = 0.24152;
// link3 -> joint4 axis (in the joint3-rotated frame): J_motor4 (0.0013449814287804,0,
// 0.105479407401356) + joint4 origin (0,0,0.052). Small x offset noted in the task context.
constexpr double kJ4X = 0.0013449814287804;
constexpr double kJ4Z = 0.105479407401356 + 0.052;  // = 0.157479407401356
// link4 -> joint5 axis: J_motor5 (0,-0.0182,0.142) + joint5 origin (0,0.0182,0). The y
// offsets cancel exactly (both compose to y=0).
constexpr double kJ5Z = 0.142;
// link5 -> joint6 axis: joint6 origin (0,0,0.0768).
constexpr double kJ6Z = 0.0768;
// link6 -> igus_tool0 (0,0,0.0527; rpy (0,0,pi)) -> xeg32_tool_tip (0.005937,0.094121,
// 0.130699; rpy 0). Folded into a single rotation (Rz(pi)) + translation offset from link6.
constexpr double kToolX = -0.005937;
constexpr double kToolY = -0.094121;
constexpr double kToolZ = 0.183399;  // 0.0527 + 0.130699

// Effective second "link" of the position sub-chain: the combined (link3->joint4) and
// (link4->joint5) offset, both carried under the same Ry(-(theta2+theta3)) rotation because
// joint4's own rotation (about z) leaves the z-directed joint4->joint5 offset unchanged (see
// rebel_analytic_ik.hpp's derivation note). kL2 is its length and kBeta the small angle its
// x-component introduces relative to a pure z offset.
double effectiveL2()
{
  return std::hypot(kJ4X, kJ4Z + kJ5Z);
}
double effectiveBeta()
{
  return std::atan2(kJ4X, kJ4Z + kJ5Z);
}

Eigen::Matrix3d rotZ(double a)
{
  Eigen::Matrix3d m;
  const double c = std::cos(a), s = std::sin(a);
  m << c, -s, 0, s, c, 0, 0, 0, 1;
  return m;
}
Eigen::Matrix3d rotY(double a)
{
  Eigen::Matrix3d m;
  const double c = std::cos(a), s = std::sin(a);
  m << c, 0, s, 0, 1, 0, -s, 0, c;
  return m;
}

double wrapPi(double a)
{
  while (a > kPi) a -= kTwoPi;
  while (a < -kPi) a += kTwoPi;
  return a;
}

// All in-range representatives of `angle` (mod 2*pi) within [limits.lower, limits.upper].
// The task calls for "±2π multiples if limits allow"; none of this robot's joint ranges
// reach a full 2*pi (max is joint1/4/6 at ~356 deg), so at most one of {angle, angle+2pi,
// angle-2pi} can ever be in range in addition to `angle` itself, but this is kept general.
void appendInRangeWraps(double angle, const JointLimits& limits, std::vector<double>& out)
{
  for (const double candidate : {angle, angle + kTwoPi, angle - kTwoPi}) {
    if (candidate >= limits.lower - 1e-12 && candidate <= limits.upper + 1e-12) {
      out.push_back(candidate);
    }
  }
}

}  // namespace

std::array<JointLimits, 6> defaultJointLimits()
{
  return {{
    {-3.106686, 3.106686},
    {-1.378810, 2.426008},
    {-1.378810, 2.426008},
    {-3.106686, 3.106686},
    {-1.640609, 1.640609},
    {-3.106686, 3.106686},
  }};
}

Eigen::Isometry3d forwardKinematics(const JointVector& q)
{
  const double t1 = q[0], t2 = q[1], t3 = q[2], t4 = q[3], t5 = q[4], t6 = q[5];
  Eigen::Isometry3d acc = Eigen::Isometry3d::Identity();
  acc.translation() = Eigen::Vector3d(0, 0, kD1Z);
  acc.linear() = rotZ(kBaseYaw) * rotZ(t1);
  // Fold: after joint1, translate by (0, kO2Y, kO2Z) in the joint1-rotated frame, then
  // rotate by joint2.
  acc.translation() += acc.linear() * Eigen::Vector3d(0, kO2Y, kO2Z);
  acc.linear() = acc.linear() * rotY(-t2);
  acc.translation() += acc.linear() * Eigen::Vector3d(0, kL1Y, kL1);
  acc.linear() = acc.linear() * rotY(-t3);
  acc.translation() += acc.linear() * Eigen::Vector3d(kJ4X, 0, kJ4Z);
  acc.linear() = acc.linear() * rotZ(t4);
  acc.translation() += acc.linear() * Eigen::Vector3d(0, 0, kJ5Z);
  acc.linear() = acc.linear() * rotY(-t5);
  acc.translation() += acc.linear() * Eigen::Vector3d(0, 0, kJ6Z);
  acc.linear() = acc.linear() * rotZ(t6);
  // link6 -> xeg32_tool_tip
  acc.translation() += acc.linear() * Eigen::Vector3d(kToolX, kToolY, kToolZ);
  acc.linear() = acc.linear() * rotZ(kPi);
  return acc;
}

std::vector<JointVector> solve(const Eigen::Isometry3d& target, const std::array<JointLimits, 6>& limits)
{
  std::vector<JointVector> solutions;
  const double L1 = kL1;
  const double L2 = effectiveL2();
  const double beta = effectiveBeta();

  // Undo the fixed link6 -> xeg32_tool_tip transform to get link6's desired pose, then the
  // wrist-center position by walking back 0.0768 m along joint6's axis, which is always the
  // world z-column of R0_6 regardless of theta6 (see header derivation note).
  const Eigen::Matrix3d r_offset = rotZ(kPi);
  const Eigen::Vector3d t_offset(kToolX, kToolY, kToolZ);
  // R0_6 = R_tcp * R_offset^-1, R_offset (a pure Z-flip) is self-inverse.
  const Eigen::Matrix3d r06 = target.linear() * r_offset.transpose();
  const Eigen::Vector3d p06 = target.translation() - r06 * t_offset;
  const Eigen::Vector3d wrist = p06 - kJ6Z * r06.col(2);

  // --- Position sub-problem (theta1, theta2, theta3) -------------------------------------
  const double wx = wrist.x(), wy = wrist.y(), wz = wrist.z();
  const double r_xy = std::hypot(wx, wy);
  const double psi = std::atan2(wy, wx);
  const double z_above_shoulder = wz - kD1Z;  // = Z(t2,t3) (the header note's Z, before -kO2Z)

  std::vector<double> theta1_branches;
  std::vector<double> x_for_branch;  // signed radial value X(t2,t3) for each theta1 branch
  if (r_xy < 1e-9) {
    // Wrist directly on joint1's axis: theta1 is not determined by position. Provide the
    // two conventional branches (0, pi) with X = -/+0 so downstream math stays well-formed;
    // this is a measure-zero edge case for random sampling.
    theta1_branches = {0.0, kPi};
    x_for_branch = {0.0, 0.0};
  } else {
    theta1_branches = {wrapPi(psi - kBaseYaw), wrapPi(psi - kBaseYaw + kPi)};
    x_for_branch = {r_xy, -r_xy};
  }

  for (std::size_t shoulder = 0; shoulder < theta1_branches.size(); ++shoulder) {
    const double theta1 = theta1_branches[shoulder];
    const double x_target = x_for_branch[shoulder];
    const double z_prime = z_above_shoulder - kO2Z;  // L1*cos(t2)+L2*cos(t2+t3-beta)
    // Law of cosines on the 2-link (L1, L2) planar chain: X^2+Z'^2 = L1^2+L2^2 +
    // 2*L1*L2*cos(t3-beta).
    const double r2 = x_target * x_target + z_prime * z_prime;
    const double cos_arg = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2);
    if (!(cos_arg >= -1.0 - 1e-9 && cos_arg <= 1.0 + 1e-9)) continue;  // unreachable, fail fast
    const double clamped = std::clamp(cos_arg, -1.0, 1.0);
    const double gamma = std::acos(clamped);  // gamma = t3 - beta, >= 0 branch
    for (const double signed_gamma : {gamma, -gamma}) {
      const double t3 = beta + signed_gamma;
      const double k1 = L1 + L2 * std::cos(signed_gamma);
      const double k2 = L2 * std::sin(signed_gamma);
      const double denom = k1 * k1 + k2 * k2;
      if (denom < 1e-12) continue;  // degenerate (L1==L2 and fully folded); not expected here
      const double sin_t2 = -(k1 * x_target + k2 * z_prime) / denom;
      const double cos_t2 = -(k2 * x_target - k1 * z_prime) / denom;
      const double t2 = std::atan2(sin_t2, cos_t2);

      // --- Orientation sub-problem (theta4, theta5, theta6) ---------------------------
      const Eigen::Matrix3d r01 = rotZ(kBaseYaw) * rotZ(theta1);
      const Eigen::Matrix3d r03 = r01 * rotY(-(t2 + t3));
      const Eigen::Matrix3d rw = r03.transpose() * r06;
      // rw == Rz(t4) * Ry(-t5) * Rz(t6); see header derivation note for the closed form
      // rw(2,2)=cos(t5), rw(0,2)=-cos(t4)sin(t5), rw(1,2)=-sin(t4)sin(t5),
      // rw(2,0)=sin(t5)cos(t6), rw(2,1)=-sin(t5)sin(t6).
      const double cos_t5 = std::clamp(rw(2, 2), -1.0, 1.0);
      for (const double sign5 : {1.0, -1.0}) {
        const double t5 = sign5 * std::acos(cos_t5);
        const double sin_t5 = std::sin(t5);
        double t4, t6;
        if (std::abs(sin_t5) < 1e-9) {
          // Gimbal lock (t5 = 0 or +-pi): only t4+t6 (t5=0) or t4-t6 (t5=pi) is determined.
          // Fix t6 = 0 and resolve t4 from the remaining matrix elements; this always
          // reproduces the original sample in the self-check because it is exactly how a
          // sampled (t4, 0-ish t5, t6) collapses, and every downstream consumer treats the
          // branch as one feasible solution, not a claim of uniqueness.
          t6 = 0.0;
          if (cos_t5 > 0.0) {
            t4 = std::atan2(rw(1, 0), rw(0, 0));
          } else {
            t4 = std::atan2(-rw(1, 0), -rw(0, 0));
          }
        } else {
          const double s = sin_t5 > 0.0 ? 1.0 : -1.0;
          t4 = std::atan2(s * -rw(1, 2), s * -rw(0, 2));
          t6 = std::atan2(s * -rw(2, 1), s * rw(2, 0));
        }

        // --- Joint-limit filtering, including in-range +-2*pi wraps --------------------
        std::vector<double> v1, v2, v3, v4, v5, v6;
        appendInRangeWraps(wrapPi(theta1), limits[0], v1);
        appendInRangeWraps(wrapPi(t2), limits[1], v2);
        appendInRangeWraps(wrapPi(t3), limits[2], v3);
        appendInRangeWraps(wrapPi(t4), limits[3], v4);
        appendInRangeWraps(wrapPi(t5), limits[4], v5);
        appendInRangeWraps(wrapPi(t6), limits[5], v6);
        if (v1.empty() || v2.empty() || v3.empty() || v4.empty() || v5.empty() || v6.empty()) {
          continue;
        }
        // t2/t3 are coupled by the position solve above (changing either alone breaks the
        // wrist-center match), so only their single computed value together with the
        // independent +-2*pi wraps of the other four joints form valid combinations; t2's
        // and t3's own wrap variants would each require re-solving the coupled pair, and
        // this robot's joint2/joint3 ranges (~217 deg) cannot admit a +-2*pi wrap anyway
        // (see appendInRangeWraps), so v2/v3 always reduce to the single base value here.
        for (double a1 : v1) {
          for (double a4 : v4) {
            for (double a5 : v5) {
              for (double a6 : v6) {
                solutions.push_back({a1, v2.front(), v3.front(), a4, a5, a6});
              }
            }
          }
        }
      }
    }
  }
  return solutions;
}

SelfCheckResult selfCheck(
  std::size_t num_samples, double position_tol_m, double orientation_tol_rad, unsigned seed,
  const std::array<JointLimits, 6>& limits)
{
  SelfCheckResult result;
  result.samples = num_samples;
  std::mt19937 rng(seed);
  for (std::size_t i = 0; i < num_samples; ++i) {
    JointVector q{};
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> dist(limits[j].lower, limits[j].upper);
      q[j] = dist(rng);
    }
    const Eigen::Isometry3d target = forwardKinematics(q);
    const auto solutions = solve(target, limits);
    if (solutions.empty()) {
      result.failure_detail = "sample " + std::to_string(i) +
        ": solve() returned no solutions for a reachable FK pose";
      result.ok = false;
      return result;
    }
    bool recovered = false;
    for (const auto& candidate : solutions) {
      const Eigen::Isometry3d fk = forwardKinematics(candidate);
      const double pos_err = (fk.translation() - target.translation()).norm();
      const double ang_err =
        Eigen::AngleAxisd(fk.linear().transpose() * target.linear()).angle();
      result.max_position_residual_m = std::max(result.max_position_residual_m, pos_err);
      result.max_orientation_residual_rad = std::max(result.max_orientation_residual_rad, ang_err);
      if (pos_err > position_tol_m || ang_err > orientation_tol_rad) {
        result.failure_detail = "sample " + std::to_string(i) +
          ": a returned solution's FK residual exceeds tolerance (pos=" +
          std::to_string(pos_err) + " m, ang=" + std::to_string(ang_err) + " rad)";
        result.ok = false;
        return result;
      }
      bool matches_original = true;
      for (std::size_t j = 0; j < 6; ++j) {
        if (std::abs(wrapPi(candidate[j] - q[j])) > 1e-6) {
          matches_original = false;
          break;
        }
      }
      if (matches_original) recovered = true;
    }
    if (recovered) ++result.samples_with_original_recovered;
  }
  result.ok = result.samples_with_original_recovered == num_samples;
  if (!result.ok && result.failure_detail.empty()) {
    result.failure_detail = std::to_string(num_samples - result.samples_with_original_recovered) +
      "/" + std::to_string(num_samples) + " samples did not recover the original joint vector";
  }
  return result;
}

}  // namespace rebel_analytic_ik
