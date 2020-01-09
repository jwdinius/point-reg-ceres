//! c/c++ headers
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include <streambuf>
#include <set>
#include <utility>
//! dependency headers
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "ceres/ceres.h"

using json = nlohmann::json;

/**
 * constants for test
 */
static const std::string data_file("../data/registration-data.json");

struct Config {
  double epsilon, pairwise_dist_threshold;
};

using correspondences_t = std::set<size_t>;

double consistency(Eigen::Vector3d const & si, Eigen::Vector3d const & tj,
    Eigen::Vector3d const & sk, Eigen::Vector3d const & tl) noexcept {
  Eigen::Vector3d const si_to_sk = si - sk;
  Eigen::Vector3d const tj_to_tl = tj - tl;
  return std::abs(si_to_sk.norm() - tj_to_tl.norm());
}

Eigen::MatrixXd generate_weight_tensor(Eigen::MatrixXd const & source_pts,
    Eigen::MatrixXd const & target_pts, Config const & config) noexcept {
  int const N = source_pts.cols();  // TODO(jwd) - make check that source and target are same size
  Eigen::MatrixXd weights = Eigen::MatrixXd::Zero((N+1)*N, (N+1)*N);
  auto const & eps = config.epsilon;
  auto const & pw_thresh = config.pairwise_dist_threshold;
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j) {
      for (auto k = 0; k < N; ++k) {
        for (auto l = 0; l < N; ++l) {
          if (i != k && j != l) {
            Eigen::Vector3d const si = source_pts.col(i);
            Eigen::Vector3d const tj = target_pts.col(j);
            Eigen::Vector3d const sk = source_pts.col(k);
            Eigen::Vector3d const tl = target_pts.col(l);
            double const c = consistency(si, tj, sk, tl);
            if (c < eps && (si - sk).norm() > pw_thresh) {
              weights(i*N+j, k*N+l) = std::exp(-c);
            }
          }
        }
      }
    }
  }
  return weights;
}

struct CostFunctor {
  explicit CostFunctor(Eigen::MatrixXd const U) : U_(U) { }
  bool operator()(double const* const* x, double* res) const {
    const double *x_deref = x[0];
    Eigen::Map<const Eigen::VectorXd> vec_x(x_deref, U_.cols() /* = U_.rows() */);
    Eigen::VectorXd weighted_vec_x = U_ * vec_x;
    res[0] = weighted_vec_x.norm();
    return true;
  }

  Eigen::MatrixXd U_;
};

int main() {
  //! load unit test data from json
  //! NOTE: this test data was generated without adding noise
  std::ifstream ifs(data_file);
  std::string json_str = std::string((std::istreambuf_iterator<char>(ifs)),
      std::istreambuf_iterator<char>());
  json json_data = json::parse(json_str);

  //! setup configuration struct for test
  Config config;
  config.epsilon = 0.1;
  config.pairwise_dist_threshold = 0.1;

  //! source pts
  auto const rows_S = json_data["source_pts"].size();
  auto const cols_S = json_data["source_pts"][0].size();
  size_t i = 0;
  Eigen::MatrixXd src_pts(rows_S, cols_S);
  std::cout << "Source Cloud Size: (" << rows_S << ", " << cols_S << ")" << std::endl;
  for (auto const & it : json_data["source_pts"]) {
    size_t j = 0;
    for (auto const & jt : it) {
      src_pts(i, j) = static_cast<double>(jt);
      ++j;
    }
    ++i;
  }

  //! target pts
  auto const rows_T = json_data["target_pts"].size();
  auto const cols_T = json_data["target_pts"][0].size();
  i = 0;
  Eigen::MatrixXd tgt_pts(rows_T, cols_T);
  std::cout << "Target Cloud Size: (" << rows_T << ", " << cols_T << ")" << std::endl;
  for (auto const & it : json_data["target_pts"]) {
    size_t j = 0;
    for (auto const & jt : it) {
      tgt_pts(i, j) = static_cast<double>(jt);
      ++j;
    }
    ++i;
  }

  correspondences_t _corrs = {};
  for (auto const & it : json_data["correspondences"]) {
    _corrs.insert(static_cast<size_t>(it));
  }

  //! SETUP THE CERES PROBLEM HERE!!
  //! first, build weighting matrix A and then take Chol decomp
  Eigen::MatrixXd A = generate_weight_tensor(src_pts, tgt_pts, config);
  //! compute Cholesky decomp: A = U.transpose() * U
  Eigen::LDLT<Eigen::MatrixXd> ldlt;
  ldlt.compute(A);
  Eigen::MatrixXd U = ldlt.matrixU();
  CostFunctor *C = new CostFunctor(U);

  //! TEST 1: make sure that CostFunctor::operator() overload works correctly
  Eigen::VectorXd v(U.rows());
  v.setRandom();
  double res[1];
  const double *vec_ptr = &v(0);
  (*C)(&vec_ptr, res);
  bool test_pass = res[0] - (U*v).norm() < std::numeric_limits<double>::epsilon();
  if (test_pass) {
    std::cout << "Overload test passed!!" << std::endl;
  } else {
    std::cout << "Overload test failed!!" << std::endl;
  }
  
  ceres::Problem problem;
  auto const & N = cols_S; /* == cols_T */
  double x[(N+1)*N];

  ceres::DynamicNumericDiffCostFunction<CostFunctor> *cost_fcn = new ceres::DynamicNumericDiffCostFunction<CostFunctor>(new CostFunctor(std::move(U)));
  cost_fcn->AddParameterBlock((N+1)*N);
  cost_fcn->SetNumResiduals(1);
  problem.AddResidualBlock(cost_fcn, nullptr /* use squared loss */, x);
  for (size_t i = 0; i < (N+1)*N; ++i) {
    problem.SetParameterLowerBound(x, i, 0.0);
    problem.SetParameterUpperBound(x, i, 1.0);
  }

  // TODO(jwd) - add residual blocks to make soft constraints out of hard constraints
  // problem.AddResidualBlock(...) for sum over i = 1
  // problem.AddResidualBlock(...) for sum over j = 1
  // problem.AddResidualBlock(...) for sum over i=N+1 = N-m (m is desired number of correspondences)
  //! now solve it!!
  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  opt.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summ;
  ceres::Solve(opt, &problem, &summ);
  std::cout << summ.FullReport() << std::endl;
  
  //! cleanup
  delete C;
  return 0;
}
