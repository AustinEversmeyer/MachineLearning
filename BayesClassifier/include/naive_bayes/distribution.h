#pragma once
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace naive_bayes {

class FeatureDistribution {
 public:
  virtual ~FeatureDistribution() {}
  virtual double LogPdf(double x) const = 0;
  virtual double Sample(std::mt19937& rng) const = 0;
  virtual std::string TypeName() const = 0;
  virtual std::vector<double> Params() const = 0;
  double Pdf(double x) const { return std::exp(LogPdf(x)); }
};

}  // namespace naive_bayes
