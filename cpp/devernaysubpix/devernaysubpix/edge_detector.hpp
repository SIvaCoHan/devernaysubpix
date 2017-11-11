#pragma once

#include "curvepoint.hpp"
#include "linkmap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv/cv.hpp>
#include <stdexcept>

namespace EdgeDetector {

struct DerivativeGaussianKernels {
  cv::Mat horz;
  cv::Mat vert;
};

struct PartialImageGradients {
  cv::Mat horz;
  cv::Mat vert;

  template <typename T>
  CurvePoint inline at(int const row, int const col) const {
    return CurvePoint(horz.at<float>(row, col), vert.at<float>(row, col),
                      false);
  }
  template <typename T> CurvePoint inline at(CurvePoint const &p) const {
    return at<T>(static_cast<int>(p.y), static_cast<int>(p.x));
  }

  template <typename T> inline T magnitude(CurvePoint const &p) const {
    int const row = static_cast<int>(p.y);
    int const col = static_cast<int>(p.x);
    T const dx = horz.at<T>(row, col);
    T const dy = vert.at<T>(row, col);
    return std::hypot(dx, dy);
  }

  cv::Mat inline magnitude() const {
    cv::Mat magn;
    cv::magnitude(horz, vert, magn);
    return magn;
  }

  cv::Mat inline threshold(double min_magnitude) const {
    return this->magnitude() > min_magnitude;
  }
};

inline DerivativeGaussianKernels
derivative_gaussian_kernels(double sigma, uint max_ksize = 64) {
  int const min_ksize = 3; // Pixels
  int const ksize = std::max<int>(
      min_ksize, static_cast<int>(std::min<double>(
                     2 * (((sigma - 0.8) / 0.3) + 1) + 1, max_ksize)));
  cv::Mat gauss_kernel_horz = cv::getGaussianKernel(ksize, sigma, CV_32F);
  cv::Mat gauss_kernel_horz_deriv;
  cv::Mat diff_kernel{0.5, 0.0, -0.5};
  // Convolving the gaussian kernel with a sobel kernel results in a first-order
  // Gaussian derivative filter
  cv::filter2D(gauss_kernel_horz, gauss_kernel_horz_deriv, -1, diff_kernel,
               {-1, -1}, 0, cv::BORDER_ISOLATED);
  cv::Mat gauss_kernel_vert_deriv = gauss_kernel_horz_deriv.t();
  return DerivativeGaussianKernels{gauss_kernel_horz_deriv,
                                   gauss_kernel_vert_deriv};
}

inline PartialImageGradients image_gradient(cv::Mat image, double sigma) {
  auto const kernels = derivative_gaussian_kernels(sigma);
  cv::Mat grad_horz;
  cv::Mat grad_vert;
  cv::filter2D(image, grad_horz, CV_32F, kernels.horz);
  cv::filter2D(image, grad_vert, CV_32F, kernels.vert);
  return PartialImageGradients{grad_horz, grad_vert};
}

struct PossibleCurvePoint {
  bool isLocalExtremum;
  CurvePoint point;
};

inline PossibleCurvePoint
compute_single_edge_point(PartialImageGradients const &gradients, int row,
                          int col) {

  auto const mag = [&gradients](int row, int col) -> float {
    return std::hypotf(gradients.horz.at<float>(row, col),
                       gradients.vert.at<float>(row, col));
  };

  float const center_mag = mag(row, col);
  float const left_mag = mag(row, col - 1);
  float const right_mag = mag(row, col + 1);
  float const top_mag = mag(row - 1, col);
  float const bottom_mag = mag(row + 1, col);

  float const abs_gx = std::abs(gradients.horz.at<float>(row, col));
  float const abs_gy = std::abs(gradients.vert.at<float>(row, col));

  int theta_x = 0;
  int theta_y = 0;
  if ((left_mag < center_mag) && (center_mag >= right_mag) &&
      (abs_gx >= abs_gy)) {
    theta_x = 1;
  } else if ((top_mag < center_mag) && (center_mag >= bottom_mag) &&
             (abs_gx <= abs_gy)) {
    theta_y = 1;
  }
  if (theta_x || theta_y) {
    float const a = mag(row - theta_y, col - theta_x);
    float const b = mag(row, col);
    float const c = mag(row + theta_y, col + theta_x);
    float const lamda = (a - c) / (2 * (a - (2 * b) + c));
    /*if (lamda>1 || lamda < -1) {
        return PossibleCurvePoint{false, {}};
    }*/
    float const ex = col + lamda * theta_x;
    float const ey = row + lamda * theta_y;
    return PossibleCurvePoint{true, CurvePoint(ex, ey, false)};
  } else {
    return PossibleCurvePoint{false, {}};
  }
}

inline std::vector<CurvePoint>
compute_edge_points(PartialImageGradients gradients, cv::Mat mask) {

  if (gradients.horz.size() != gradients.vert.size()) {
    throw std::runtime_error("Image gradients differ in size");
  }

  if (gradients.horz.size() != mask.size()) {
    throw std::runtime_error("Mask and image gradients differ in size");
  }

  if (gradients.horz.channels() != 1) {
    throw std::runtime_error("Horizontal image must have single channel");
  }

  if (gradients.vert.channels() != 1) {
    throw std::runtime_error("Vertical image must have single channel");
  }

  int const rows = gradients.horz.rows;
  int const columns = gradients.horz.cols;

  if ((rows < 3) || (columns < 3)) {
    throw std::runtime_error("Input must be at least 3x3 pixels big");
  }

  std::vector<CurvePoint> edges;

  // TODO: Parallelize
  for (int row = 1; row < (rows - 1); ++row) {
    for (int col = 1; col < (columns - 1); ++col) {
      if (mask.at<uint8_t>(row, col)) {
        PossibleCurvePoint p = compute_single_edge_point(gradients, row, col);
        if (p.isLocalExtremum) {
          if (p.point.x < 0 || p.point.y < 0) {
            throw std::runtime_error("BELOW 0");
          }
          if (std::isnan(p.point.x) || std::isnan(p.point.y)) {
            throw std::runtime_error("NAN");
          }
          edges.push_back(p.point);
        }
      }
    }
  }

  return edges;
}

struct NearestNeighborhoodPoints {
  bool forward_valid;
  bool backward_valid;
  CurvePoint forward;
  CurvePoint backward;
};

CurvePoint inline perpendicular_vec(CurvePoint const &p) {
  return CurvePoint(p.y, -p.x, p.valid);
}

/**
 * @brief Computes dot (scalar) product between two points
 * @param p1 Left-hand point
 * @param p2 Right-hand point
 * @return Dot-product
 */
float inline dot(CurvePoint const &p1, CurvePoint const &p2) {
  return (p1.x * p2.x) + (p1.y * p2.y);
}

NearestNeighborhoodPoints inline find_nearest_forward_and_backward(
    std::vector<CurvePoint> const &neighborhood,
    PartialImageGradients const &gradients, CurvePoint const &reference) {

  // Default: invalid points
  NearestNeighborhoodPoints out{false, false, CurvePoint(), CurvePoint()};

  if (neighborhood.size() == 0) {
    return out;
  }

  // Gradient values at g(e.x, e.y)
  CurvePoint ge(gradients.at<float>(reference));

  float min_distance_forward = std::numeric_limits<float>::max();
  float min_distance_backward = std::numeric_limits<float>::max();
  for (auto const &p : neighborhood) {
    // Gradient values at g(n.x, n.y)
    CurvePoint gn(gradients.at<float>(p));
    float distance = reference.distance(p);
    auto const angle_lower_than_90_degrees = dot(ge, gn) > 0;
    if (!angle_lower_than_90_degrees) {
      continue;
    }

    auto const angle_perp = dot(p - reference, perpendicular_vec(ge));
    bool const is_forward = angle_perp > 0;
    bool const is_backward = angle_perp < 0;
    if (is_forward) {
      out.forward_valid = true;
      if (distance < min_distance_forward) {
        out.forward = p;
        min_distance_forward = distance;
      }
    } else if (is_backward) {
      out.backward_valid = true;
      if (distance < min_distance_backward) {
        out.backward = p;
        min_distance_backward = distance;
      }
    }
  }
  return out;
}

std::vector<CurvePoint> inline get_neighborhood(
    std::vector<CurvePoint> const &edges, CurvePoint const &p,
    float max_distance) {
  // TODO: Optimize by precomputing neighborhood and put into 2d array
  std::vector<CurvePoint> hood;
  for (auto const &edge : edges) {
    if (std::abs(edge.x - p.x) <= max_distance &&
        std::abs(edge.y - p.y) <= max_distance) {
      hood.push_back(edge);
    }
  }
  return hood;
}

LinkMap inline chain_edge_points(std::vector<CurvePoint> const &edges,
                                 PartialImageGradients const &gradients) {
  LinkMap links;
  for (auto const &e : edges) {
    auto const neighborhood = get_neighborhood(edges, e, 2);
    auto const nearest =
        find_nearest_forward_and_backward(neighborhood, gradients, e);
    if (nearest.forward_valid) {
      auto const &f = nearest.forward;
      // L6
      if (!links.hasRight(f) ||
          (e.distance(f) < links.getByRight(f).first.distance(f))) {
        links.unlinkByRight(f);
        links.unlinkByLeft(e);
        links.link(e, f);
      }
    }
    if (nearest.backward_valid) {
      auto const &b = nearest.backward;
      // L10
      if (!links.hasLeft(b) ||
          (b.distance(e) < b.distance(links.getByLeft(b).second))) {
        links.unlinkByLeft(b);
        links.unlinkByRight(e);
        links.link(b, e);
      }
    }
  }
  return links;
}

using Chain = std::vector<CurvePoint>;
using Chains = std::vector<Chain>;

Chains inline thresholds_with_hysteresis(std::vector<CurvePoint> &edges,
                                         LinkMap &links,
                                         PartialImageGradients const &grads,
                                         float const high_threshold,
                                         float const low_threshold) {
  Chains chains;
  // Ensure all edges are invalid at first so all edges are considered for
  // chaining
  for (auto &e : edges) {
    e.valid = false;
  }

  for (auto &e : edges) {
    if (!e.valid && grads.magnitude<float>(e) >= high_threshold) {
      Chain forward;
      Chain backward;
      e.valid = true;
      auto f = e;
      while (links.hasLeft(f) && !(links.getByLeft(f).second.valid) &&
             (grads.magnitude<float>(links.getByLeft(f).second) >=
              low_threshold)) {
        auto n = links.getByLeft(f).second;
        n.valid = true;
        links.replace(f, n);
        f = n;
        forward.push_back(f);
      }
      auto b = e;
      while (links.hasRight(b) && !(links.getByRight(b).first.valid) &&
             (grads.magnitude<float>(links.getByRight(b).first) >=
              low_threshold)) {
        auto n = links.getByRight(b).first;
        n.valid = true;
        links.replace(n, b);
        b = n;
        backward.insert(backward.begin(), b);
      }
      Chain chain;
      std::copy(backward.begin(), backward.end(), std::back_inserter(chain));
      chain.push_back(e);
      std::copy(forward.begin(), forward.end(), std::back_inserter(chain));
      if (chain.size() > 1) {
        chains.push_back(std::move(chain));
      }
    }
  }
  return chains;
}

} // namespace EdgeDetector
