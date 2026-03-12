/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/LieGroup.h, src/cvo/LieGroup.cpp (no 1:1 mapping claimed)
 *
 * This GCVO repository is NOT a verbatim copy of RKHS_BA. It includes substantial modifications,
 * refactors, and additional original content (e.g., solver/optimization changes and new utilities).
 *
 * References:
 * - RKHS_BA paper: R. Zhang et al., IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3593521
 * - GCVO paper: R. Zhang et al., CVPR 2026 (see repo README for details)
 *
 * License: RKHS_BA is MIT-licensed (see rkhs_ba/ for the upstream LICENSE). This repo’s license is in
 * the root LICENSE file. Contact (GCVO modifications): ray.zhang@tri.global
 */

#pragma once
/*
 *  Header-only Lie group utilities extracted/refactored from original GCVO code:
 *   https://github.com/UMich-CURLY/RKHS_BA/
 *    - solver/LieGroup.h (+ .cpp implementation)
 *
 *  Refactor goals:
 *    - Header-only, independent include for GCVO
 *    - Templated on scalar type T (float/double) and storage order RC_MAJOR
 *    - Preserve original mathematical formulas/behavior
 *
 *  NOTE ON LICENSING:
 *    - If upstream GCVO has a specific license header, replace this block accordingly.
 *
 *  Date: 2026-03-09
 */

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <type_traits>
#include <algorithm>

namespace gcvo {
  namespace liegroup {
    template <typename T>
    constexpr T tolerance() {
      return static_cast<T>(1e-6);
    }

    // -----------------------
    // so(3) helpers
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 3, 3, RC_MAJOR> skew(const Eigen::Matrix<T, 3, 1>& v) {
      Eigen::Matrix<T, 3, 3, RC_MAJOR> M = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Zero();
      M << T(0),   -v[2],  v[1],
        v[2],   T(0),  -v[0],
        -v[1],   v[0],  T(0);
      return M;
    }

    template <typename T>
    inline Eigen::Matrix<T, 3, 1> unskew(const Eigen::Matrix<T, 3, 3>& M) {
      Eigen::Matrix<T, 3, 1> v;
      v << M(2,1), M(0,2), M(1,0);
      return v;
    }

    // -----------------------
    // SE(3) hat/wedge (4x4 form)
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 4, 4, RC_MAJOR> hat2(const Eigen::Matrix<T, 6, 1>& x) {
      Eigen::Matrix<T, 4, 4, RC_MAJOR> X = Eigen::Matrix<T, 4, 4, RC_MAJOR>::Zero();
      X.template block<3,3>(0,0) = skew<T, RC_MAJOR>(x.template head<3>());
      X.template block<3,1>(0,3) = x.template tail<3>();
      return X;
    }

    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 6, 1> wedge(const Eigen::Matrix<T, 4, 4, RC_MAJOR>& X) {
      Eigen::Matrix<T, 6, 1> x;
      x.template head<3>() = unskew<T>(X.template block<3,3>(0,0));
      x.template tail<3>() = X.template block<3,1>(0,3);
      return x;
    }

    // -----------------------
    // SO(3) Exp / Log
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 3, 3, RC_MAJOR> Exp_SO3(const Eigen::Matrix<T, 3, 1>& w) {
      const T theta = w.norm();
      if (theta < tolerance<T>()) {
        return Eigen::Matrix<T,3,3,RC_MAJOR>::Identity();
      }

      const auto A  = skew<T, RC_MAJOR>(w);
      const auto A2 = A * A;

      const T s = std::sin(theta);
      const T c = std::cos(theta);

      return Eigen::Matrix<T,3,3,RC_MAJOR>::Identity()
        + (s/theta) * A
        + ((T(1)-c)/(theta*theta)) * A2;
    }

    template <typename T>
    inline Eigen::Matrix<T, 3, 1> Log_SO3(const Eigen::Matrix<T, 3, 3>& R) {
      // theta = acos((trace(R)-1)/2)
      const T half = T(0.5);
      T x = (R.trace() - T(1)) * half;
      x = std::min(T(1), std::max(T(-1), x));
      const T theta = std::acos(x);

      if (theta < tolerance<T>()) {
        return Eigen::Matrix<T,3,1>::Zero();
      }

      const T s = std::sin(theta);
      return unskew<T>((theta/(T(2)*s)) * (R - R.transpose()));
    }

    // -----------------------
    // SO(3) Left Jacobian / Inverse
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 3, 3, RC_MAJOR> LeftJacobian_SO3(const Eigen::Matrix<T, 3, 1>& w) {
      const T theta = w.norm();
      if (theta < tolerance<T>()) {
        return Eigen::Matrix<T,3,3,RC_MAJOR>::Identity();
      }

      const auto A  = skew<T, RC_MAJOR>(w);
      const auto A2 = A * A;

      const T s = std::sin(theta);
      const T c = std::cos(theta);

      // I + ((1-c)/theta^2) A + ((theta-s)/theta^3) A^2
      return Eigen::Matrix<T,3,3,RC_MAJOR>::Identity()
        + ((T(1)-c)/(theta*theta)) * A
        + ((theta - s)/(theta*theta*theta)) * A2;
    }

    template <typename T>
    inline Eigen::Matrix<T, 3, 3> LeftJacobianInverse_SO3(const Eigen::Matrix<T, 3, 1>& w) {
      const T theta = w.norm();
      if (theta < tolerance<T>()) {
        return Eigen::Matrix<T,3,3>::Identity();
      }

      const Eigen::Matrix<T,3,3> A  = skew<T, Eigen::ColMajor>(w); // safe for math; storage not critical here
      const Eigen::Matrix<T,3,3> A2 = A * A;

      const T s = std::sin(theta);
      const T c = std::cos(theta);

      // Matches original GCVO expression:
      // I - 0.5*A + (1/theta^2 - (1+c)/(2*theta*s)) * A^2
      const T term = (T(1)/(theta*theta)) - ((T(1)+c)/(T(2)*theta*s));
      return Eigen::Matrix<T,3,3>::Identity() - T(0.5)*A + term*A2;
    }

    // -----------------------
    // SE(3) Left Jacobian
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 6, 6, RC_MAJOR> LeftJacobian_SE3(Eigen::Matrix<T, 6, 1>& v) {
      Eigen::Matrix<T, 6, 6, RC_MAJOR> output = Eigen::Matrix<T,6,6,RC_MAJOR>::Zero();

      Eigen::Matrix<T,3,1> Phi = v.template head<3>();
      const T phi = Phi.norm();
      Eigen::Matrix<T,3,1> Rho = v.template tail<3>();

      const Eigen::Matrix<T,3,3,RC_MAJOR> Phi_skew = skew<T, RC_MAJOR>(Phi);
      const Eigen::Matrix<T,3,3,RC_MAJOR> Rho_skew = skew<T, RC_MAJOR>(Rho);

      const Eigen::Matrix<T,3,3,RC_MAJOR> J = LeftJacobian_SO3<T, RC_MAJOR>(Phi);
      Eigen::Matrix<T,3,3,RC_MAJOR> Q = Eigen::Matrix<T,3,3,RC_MAJOR>::Zero();

      if (phi < tolerance<T>()) {
        Q = T(0.5) * Rho_skew;
      } else {
        const T phi2 = phi*phi;
        const T phi3 = phi2*phi;
        const T phi4 = phi3*phi;
        const T phi5 = phi4*phi;

        Q = T(0.5)*Rho_skew
          + (phi - std::sin(phi))/phi3 * (Phi_skew*Rho_skew + Rho_skew*Phi_skew + Phi_skew*Rho_skew*Phi_skew)
          - (T(1) - T(0.5)*phi2 - std::cos(phi))/phi4
          * (Phi_skew*Phi_skew*Rho_skew + Rho_skew*Phi_skew*Phi_skew - T(3)*Phi_skew*Rho_skew*Phi_skew)
          - T(0.5) * ((T(1) - T(0.5)*phi2 - std::cos(phi))/phi4 - T(3)*(phi - std::sin(phi) - phi3/T(6))/phi5)
          * (Phi_skew*Rho_skew*Phi_skew*Phi_skew + Phi_skew*Phi_skew*Rho_skew*Phi_skew);
      }

      output.template block<3,3>(0,0) = J;
      output.template block<3,3>(0,3) = Q;
      output.template block<3,3>(3,3) = J;
      return output;
    }

    // -----------------------
    // SE_K(3) Adjoint (needed for RightJacobian_SE3 via Adjoint*LeftJacobian)
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>
    Adjoint_SEK3(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>& X) {
      const int K = static_cast<int>(X.cols()) - 3;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> Adj;
      Adj.setZero(3 + 3*K, 3 + 3*K);

      const Eigen::Matrix<T,3,3,RC_MAJOR> R = X.template block<3,3>(0,0);
      Adj.template block<3,3>(0,0) = R;

      for (int i = 0; i < K; ++i) {
        Adj.template block<3,3>(3+3*i, 3+3*i) = R;
        Adj.template block<3,3>(3+3*i, 0) = skew<T, RC_MAJOR>(X.template block<3,1>(0,3+i)) * R;
      }
      return Adj;
    }

    // -----------------------
    // SE(3) Right Jacobian / Inverse (same structure as original code)
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 6, 6, RC_MAJOR> RightJacobian_SE3(Eigen::Matrix<T, 6, 1>& v) {
      if (v.norm() < tolerance<T>()) {
        return Eigen::Matrix<T,6,6,RC_MAJOR>::Identity();
      }
      // Jr(v) = Adjoint(exp(hat(-v))) * Jl(v)
      const auto X = hat2<T, RC_MAJOR>(-v).exp();
      return Adjoint_SEK3<T, RC_MAJOR>(X) * LeftJacobian_SE3<T, RC_MAJOR>(v);
    }

    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 6, 6, RC_MAJOR> RightJacobianInverse_SE3(Eigen::Matrix<T, 6, 1>& v) {
      if (v.norm() < tolerance<T>()) {
        return Eigen::Matrix<T,6,6,RC_MAJOR>::Identity();
      }
      return RightJacobian_SE3<T, RC_MAJOR>(v).inverse();
    }

    // -----------------------
    // SE(3) Exp / Log
    // -----------------------
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 4, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v) {
      const Eigen::Matrix<T,3,1> w = v.template head<3>();
      const Eigen::Matrix<T,3,1> u = v.template tail<3>();

      Eigen::Matrix<T, 4, 4, RC_MAJOR> X = Eigen::Matrix<T,4,4,RC_MAJOR>::Identity();
      X.template block<3,3>(0,0) = Exp_SO3<T, RC_MAJOR>(w);
      X.template block<3,1>(0,3) = LeftJacobian_SO3<T, RC_MAJOR>(w) * u;
      return X;
    }

    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 6, 1> Log_SE3(const Eigen::Matrix<T, 4, 4, RC_MAJOR>& X) {
      const Eigen::Matrix<T,3,3> R = X.template block<3,3>(0,0);
      const Eigen::Matrix<T,3,1> t = X.template block<3,1>(0,3);

      const Eigen::Matrix<T,3,1> w = Log_SO3<T>(R);
      const Eigen::Matrix<T,3,1> v = LeftJacobianInverse_SO3<T>(w) * t;

      Eigen::Matrix<T,6,1> s;
      s.template head<3>() = w;
      s.template tail<3>() = v;
      return s;
    }

    // Compatibility overload mirroring original code signature: Exp_SE3(v, is_wu)
    template <typename T, int RC_MAJOR>
    inline Eigen::Matrix<T, 3, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v, bool is_wu) {
      Eigen::Matrix<T,3,1> w, u;
      if (is_wu) { w = v.template head<3>(); u = v.template tail<3>(); }
      else       { w = v.template tail<3>(); u = v.template head<3>(); }

      Eigen::Matrix<T,3,4,RC_MAJOR> X;
      X.setZero();
      const auto R = Exp_SO3<T, RC_MAJOR>(w);
      X.template block<3,3>(0,0) = R;
      X.template block<3,1>(0,3) = LeftJacobian_SO3<T, RC_MAJOR>(w) * u;
      return X;
    }

    // Distance via || log(T) ||_F (same structure as original dist_se3)
    template <typename T, int RC_MAJOR>
    inline T dist_se3(const Eigen::Matrix<T, 3, 3, RC_MAJOR>& R,
                      const Eigen::Matrix<T, 3, 1>& t) {
      Eigen::Matrix<T, 4, 4, RC_MAJOR> X = Eigen::Matrix<T,4,4,RC_MAJOR>::Identity();
      X.template block<3,3>(0,0) = R;
      X.template block<3,1>(0,3) = t;
      return static_cast<T>(X.log().norm());
    }
  } // namespace liegroup::
}  // namespace gcvo::
