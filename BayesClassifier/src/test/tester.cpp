#include "test/tester.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>

#include "naive_bayes/naive_bayes.h"
#include "naive_bayes/gaussian.h"
#include "naive_bayes/rayleigh.h"
#include "io/model_loader.h"
#include "pipeline/pipeline_config.h"
#include "pipeline/pipeline_helpers.h"
#include "io/json.h"

namespace naive_bayes::test {

namespace {

// Helper function for floating point comparisons
void AssertAlmostEqual(double a, double b, double epsilon = 1e-5, const std::string& msg = "") {
    if (std::abs(a - b) > epsilon) {
        throw std::runtime_error("Assertion failed: " + std::to_string(a) + " != " + std::to_string(b) + " (Expected vs Actual: " + std::to_string(b) + " vs " + std::to_string(a) + ") " + msg);
    }
}

void TestDistributions() {
    std::cout << "[Test] FeatureDistribution Core Logic..." << std::endl;
    
    // --- Gaussian Test (Mean 0, Sigma 1) ---
    naive_bayes::Gaussian g(0.0, 1.0);
    // PDF at 0 should be 1/sqrt(2pi) approx 0.398942
    double pdf_at_0 = std::exp(g.LogPdf(0.0));
    AssertAlmostEqual(pdf_at_0, 0.398942, 1e-6, "Gaussian PDF at 0.0");
    
    // PDF at 1 should be 0.398942 * exp(-0.5) approx 0.241970
    double pdf_at_1 = std::exp(g.LogPdf(1.0));
    AssertAlmostEqual(pdf_at_1, 0.241970, 1e-6, "Gaussian PDF at 1.0");

    // --- Rayleigh Test (Sigma 1) ---
    naive_bayes::Rayleigh r(1.0);
    // PDF at 1: (1/1^2) * 1 * exp(-1^2 / 2) = exp(-0.5) approx 0.606530
    double r_pdf_at_1 = std::exp(r.LogPdf(1.0));
    AssertAlmostEqual(r_pdf_at_1, 0.606530, 1e-6, "Rayleigh PDF at 1.0");
    
    // PDF at 0 should be 0 (log should be -inf)
    double log_val = r.LogPdf(0.0);
    if (log_val != -std::numeric_limits<double>::infinity()) {
         throw std::runtime_error("Rayleigh PDF at 0.0 should be -inf");
    }

    // --- Test edge cases for distribution initialization ---
    bool caught_exception = false;
    try {
        naive_bayes::Gaussian g_bad(0.0, 0.0); // sigma <= 0
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    if (!caught_exception) throw std::runtime_error("Failed to catch invalid_argument for Gaussian sigma <= 0");
    
    caught_exception = false;
    try {
        naive_bayes::Rayleigh r_bad(-1.0); // sigma <= 0
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    if (!caught_exception) throw std::runtime_error("Failed to catch invalid_argument for Rayleigh sigma <= 0");
}

void TestLogSumExpLogic() {
    std::cout << "[Test] NaiveBayes Log-Space and Numerical Stability..." << std::endl;
    
    // --- Setup for Log-Sum-Exp Test ---
    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);
    
    // Model A: Prior 0.5, Feature Gaussian(0,1)
    naive_bayes::ClassDefinition modelA;
    modelA.name = "A";
    modelA.prior = 0.5;
    modelA.feature_names = {"x"};
    modelA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));
    
    // Model B: Prior 0.5, Feature Gaussian(10,1) -> High separation
    naive_bayes::ClassDefinition modelB;
    modelB.name = "B";
    modelB.prior = 0.5;
    modelB.feature_names = {"x"};
    modelB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(10.0, 1.0));
    
    clf.AddClassDefinition(std::move(modelA));
    clf.AddClassDefinition(std::move(modelB));
    
    // --- Test 1: Prediction at the mean of A (x=0) ---
    // P(A) should be very close to 1.0
    std::vector<double> input_A = {0.0};
    auto probs = clf.PredictPosteriors(input_A);
    
    double prob_A = 0.0;
    for(const auto& p : probs) {
        if(p.first == "A") prob_A = p.second;
    }
    AssertAlmostEqual(prob_A, 1.0, 1e-8, "Class A probability at x=0");

    // --- Test 2: Prediction exactly between means (x=5) ---
    // P(A) and P(B) should be exactly 0.5
    std::vector<double> input_mid = {5.0};
    probs = clf.PredictPosteriors(input_mid);
    
    prob_A = 0.0;
    double prob_B = 0.0;
    for(const auto& p : probs) {
        if(p.first == "A") prob_A = p.second;
        if(p.first == "B") prob_B = p.second;
    }
    AssertAlmostEqual(prob_A, 0.5, 1e-8, "Class A probability at x=5");
    AssertAlmostEqual(prob_B, 0.5, 1e-8, "Class B probability at x=5");
}

void TestLayoutMapping() {
    std::cout << "[Test] Layout alignment with model feature ordering..." << std::endl;

    std::vector<std::string> model_features = {"temp", "pressure", "humidity"};

    // Layout inherits model ordering when unspecified
    naive_bayes::pipeline::LayoutConfig empty_layout;
    naive_bayes::pipeline::AlignLayoutWithModel(empty_layout, model_features);
    if (empty_layout.feature_fields != model_features) {
        throw std::runtime_error("Empty layout did not adopt model feature order");
    }

    // Layout with same features in different order should be reordered to model order
    naive_bayes::pipeline::LayoutConfig shuffled_layout;
    shuffled_layout.feature_fields = {"pressure", "temp", "humidity"};
    naive_bayes::pipeline::AlignLayoutWithModel(shuffled_layout, model_features);
    if (shuffled_layout.feature_fields != model_features) {
        throw std::runtime_error("Layout features were not realigned to model order");
    }

    // Mismatched feature sets should throw
    naive_bayes::pipeline::LayoutConfig bad_layout;
    bad_layout.feature_fields = {"temp", "pressure"};
    bool threw = false;
    try {
        naive_bayes::pipeline::AlignLayoutWithModel(bad_layout, model_features);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("Expected AlignLayoutWithModel to throw on mismatched features");
    }

    std::cout << "[Test] Layout alignment validated." << std::endl;
}

void TestLiveJsonPredictionLoop() {
    std::cout << "[Test] Live JSON single prediction helper..." << std::endl;

    naive_bayes::pipeline::LayoutConfig layout;
    layout.feature_fields = {"x"};

    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);

    naive_bayes::ClassDefinition classA;
    classA.name = "A";
    classA.prior = 0.5;
    classA.feature_names = {"x"};
    classA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));

    naive_bayes::ClassDefinition classB;
    classB.name = "B";
    classB.prior = 0.5;
    classB.feature_names = {"x"};
    classB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(5.0, 1.0));

    clf.AddClassDefinition(std::move(classA));
    clf.AddClassDefinition(std::move(classB));

    auto MakeEvent = [](double x_val) {
        return naive_bayes::io::Json::object({{"x", x_val}});
    };

    std::vector<naive_bayes::io::Json> events = {MakeEvent(0.0), MakeEvent(0.5), MakeEvent(6.0)};
    std::vector<std::string> expected = {"A", "A", "B"};

    for (std::size_t i = 0; i < events.size(); ++i) {
        naive_bayes::pipeline::SinglePrediction result = naive_bayes::pipeline::PredictResultFromJsonObject(clf, layout, events[i]);
        if (result.predicted_class != expected[i]) {
            throw std::runtime_error("Unexpected predicted class at event " + std::to_string(i));
        }
    }

    naive_bayes::io::Json bad_event;
    naive_bayes::pipeline::SinglePrediction missing_result = naive_bayes::pipeline::PredictResultFromJsonObject(clf, layout, bad_event);
    AssertAlmostEqual(missing_result.predicted_prob, 0.5, 1e-8, "Missing feature should fall back to prior probability");

    std::cout << "[Test] Live JSON helper proved mapping and error handling." << std::endl;
}

void TestMissingFeatureInference() {
    std::cout << "[Test] Missing feature values are skipped during inference..." << std::endl;

    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);

    naive_bayes::ClassDefinition classA;
    classA.name = "A";
    classA.prior = 0.5;
    classA.feature_names = {"length", "height"};
    classA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));
    classA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));

    naive_bayes::ClassDefinition classB;
    classB.name = "B";
    classB.prior = 0.5;
    classB.feature_names = {"length", "height"};
    classB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(5.0, 1.0));
    classB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(5.0, 1.0));

    clf.AddClassDefinition(std::move(classA));
    clf.AddClassDefinition(std::move(classB));

    std::vector<double> only_length = {0.0, std::numeric_limits<double>::quiet_NaN()};
    auto probs = clf.PredictPosteriors(only_length);
    std::string argmax;
    double best = -1.0;
    for (const auto& p : probs) {
        if (p.second > best) {
            best = p.second;
            argmax = p.first;
        }
    }
    if (argmax != "A") {
        throw std::runtime_error("Expected class A to win when only length favors it");
    }

    std::vector<double> only_height = {std::numeric_limits<double>::quiet_NaN(), 5.0};
    probs = clf.PredictPosteriors(only_height);
    argmax.clear();
    best = -1.0;
    for (const auto& p : probs) {
        if (p.second > best) {
            best = p.second;
            argmax = p.first;
        }
    }
    if (argmax != "B") {
        throw std::runtime_error("Expected class B to win when only height favors it");
    }

    naive_bayes::pipeline::LayoutConfig layout;
    layout.feature_fields = {"length", "height"};
    naive_bayes::io::Json json_obj = naive_bayes::io::Json::object({{"length", 5.0}});
    // intentionally omit "height"
    naive_bayes::pipeline::SinglePrediction result = naive_bayes::pipeline::PredictResultFromJsonObject(clf, layout, json_obj);
    if (result.predicted_class != "B") {
        throw std::runtime_error("JSON with missing height should still predict class B based on length only");
    }

    std::cout << "[Test] Missing feature inference validated." << std::endl;
}

void TestFeatureUnionAndMissingSlots() {
    std::cout << "[Test] Feature union with class-specific missing features..." << std::endl;

    // Build a temporary model config where ClassB omits feature f2.
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_feature_union_test.json";
    std::ofstream out(tmp);
    out << R"({
      "computation_mode": "log",
      "classes": [
        {
          "name": "ClassA",
          "prior": 0.5,
          "features": [
            { "name": "f1", "type": "gaussian", "params": { "mean": 0.0, "sigma": 1.0 } },
            { "name": "f2", "type": "gaussian", "params": { "mean": 0.0, "sigma": 0.1 } }
          ]
        },
        {
          "name": "ClassB",
          "prior": 0.5,
          "features": [
            { "name": "f1", "type": "gaussian", "params": { "mean": 0.0, "sigma": 1.0 } }
          ]
        }
      ]
    })";
    out.close();

    naive_bayes::NaiveBayes model = naive_bayes::io::LoadModelConfiguration(tmp);
    if (model.FeatureDim() != 2) {
        throw std::runtime_error("Expected feature dim 2 from feature union");
    }

    // f2 is highly discriminative for ClassA near 0; ClassB ignores f2 entirely.
    // At f2 = 10, ClassA likelihood plummets; ClassB should win despite equal priors.
    std::vector<double> features = {0.0, 10.0};
    auto probs = model.PredictPosteriors(features);
    std::string argmax;
    double best = -1.0;
    for (const auto& p : probs) {
        if (p.second > best) {
            best = p.second;
            argmax = p.first;
        }
    }
    if (argmax != "ClassB") {
        throw std::runtime_error("Expected ClassB to win when ClassA models penalizing f2 and ClassB skips it");
    }
}

void TestDuplicateFeaturesRejected() {
    std::cout << "[Test] Duplicate feature names are rejected..." << std::endl;

    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_duplicate_feature_test.json";
    std::ofstream out(tmp);
    out << R"({
      "classes": [
        {
          "name": "ClassA",
          "prior": 0.5,
          "features": [
            { "name": "f1", "type": "gaussian", "params": { "mean": 0.0, "sigma": 1.0 } },
            { "name": "f1", "type": "gaussian", "params": { "mean": 1.0, "sigma": 1.0 } }
          ]
        }
      ]
    })";
    out.close();

    bool threw = false;
    try {
        (void)naive_bayes::io::LoadModelConfiguration(tmp);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("Expected duplicate features in a class to be rejected");
    }
}

void TestClassGroupPosteriors() {
    std::cout << "[Test] Class group posteriors sum correctly..." << std::endl;

    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);

    auto MakeClass = [](const std::string& name) {
        naive_bayes::ClassDefinition cls;
        cls.name = name;
        cls.prior = 0.25;
        cls.feature_names = {"x"};
        cls.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));
        return cls;
    };

    clf.AddClassDefinition(MakeClass("DogA"));
    clf.AddClassDefinition(MakeClass("DogB"));
    clf.AddClassDefinition(MakeClass("CatA"));
    clf.AddClassDefinition(MakeClass("CatB"));

    std::vector<naive_bayes::ClassGroup> groups = {
        {"dog", {"DogA", "DogB"}},
        {"cat", {"CatA", "CatB"}},
    };
    clf.SetClassGroups(groups);

    std::vector<double> features = {0.0};
    auto grouped = clf.PredictGroupedPosteriors(features);

    double prob_dog = 0.0;
    double prob_cat = 0.0;
    for (const auto& entry : grouped) {
        if (entry.first == "dog") {
            prob_dog = entry.second;
        } else if (entry.first == "cat") {
            prob_cat = entry.second;
        }
    }
    AssertAlmostEqual(prob_dog, 0.5, 1e-8, "Dog group should sum to 0.5");
    AssertAlmostEqual(prob_cat, 0.5, 1e-8, "Cat group should sum to 0.5");
}

void TestCsvGroupColumns() {
    std::cout << "[Test] CSV includes grouped columns when configured..." << std::endl;

    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);

    naive_bayes::ClassDefinition classA;
    classA.name = "A";
    classA.prior = 0.5;
    classA.feature_names = {"x"};
    classA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));

    naive_bayes::ClassDefinition classB;
    classB.name = "B";
    classB.prior = 0.5;
    classB.feature_names = {"x"};
    classB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(1.0, 1.0));

    clf.AddClassDefinition(std::move(classA));
    clf.AddClassDefinition(std::move(classB));
    clf.SetClassGroups({{"GroupA", {"A"}}, {"GroupB", {"B"}}});

    naive_bayes::pipeline::Observation obs;
    obs.timestep = 0.0;
    obs.truth_label = "A";
    obs.features = {0.0};

    std::vector<naive_bayes::pipeline::BatchPredictionRow> rows =
        naive_bayes::pipeline::RunInference(clf, {obs});

    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_group_csv_test.csv";
    naive_bayes::pipeline::WritePredictionsCsv(tmp, rows, false);

    std::ifstream input(tmp);
    if (!input) {
        throw std::runtime_error("Failed to open CSV output for grouped columns test");
    }
    std::string header;
    std::getline(input, header);

    const std::string expected_header =
        "time_ns,id,truth_label,classification_state,feature_x,predicted_class,predicted_prob,prob_A,prob_B,"
        "predicted_group,predicted_group_prob,group_prob_GroupA,group_prob_GroupB";
    if (header != expected_header) {
        throw std::runtime_error("Grouped CSV header mismatch: " + header);
    }
}

void TestCsvFeatureInputsAndIdColumns() {
    std::cout << "[Test] CSV includes id and feature input columns..." << std::endl;

    std::vector<naive_bayes::pipeline::BatchPredictionRow> rows;

    naive_bayes::pipeline::BatchPredictionRow row1;
    row1.time_ns = 10000000000LL;
    row1.id = 7;
    row1.truth_label = "truthA";
    row1.classification_state = "partial";
    row1.feature_inputs = {
        {"x", 1.25},
        {"y", std::numeric_limits<double>::quiet_NaN()},
    };
    row1.predicted_class = "A";
    row1.predicted_prob = 0.9;
    row1.probabilities = {{"A", 0.9}, {"B", 0.1}};
    rows.push_back(std::move(row1));

    naive_bayes::pipeline::BatchPredictionRow row2;
    row2.time_ns = 11000000000LL;
    row2.truth_label = "truthB";
    row2.classification_state = "full";
    row2.feature_inputs = {
        {"x", 2.5},
        {"z", 99.0},  // Deliberately omit canonical "y"; writer should emit nan.
    };
    row2.predicted_class = "B";
    row2.predicted_prob = 0.8;
    row2.probabilities = {{"A", 0.2}, {"B", 0.8}};
    rows.push_back(std::move(row2));

    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_feature_id_csv_test.csv";
    naive_bayes::pipeline::WritePredictionsCsv(tmp, rows, false);

    std::ifstream input(tmp);
    if (!input) {
        throw std::runtime_error("Failed to open CSV output for feature/id columns test");
    }

    std::string header;
    std::getline(input, header);
    const std::string expected_header =
        "time_ns,id,truth_label,classification_state,feature_x,feature_y,predicted_class,predicted_prob,prob_A,prob_B";
    if (header != expected_header) {
        throw std::runtime_error("Feature/id CSV header mismatch: " + header);
    }

    std::string line1;
    std::getline(input, line1);
    if (line1 != "10000000000,7,truthA,partial,1.250000,nan,A,0.900000,0.900000,0.100000") {
        throw std::runtime_error("Unexpected first CSV data row: " + line1);
    }

    std::string line2;
    std::getline(input, line2);
    if (line2 != "11000000000,,truthB,full,2.500000,nan,B,0.800000,0.200000,0.800000") {
        throw std::runtime_error("Unexpected second CSV data row: " + line2);
    }
}

void TestDetailedPredictionHelperAndFeatureZip() {
    std::cout << "[Test] Shared detailed prediction helper and feature zip..." << std::endl;

    naive_bayes::NaiveBayes clf(naive_bayes::ProbabilitySpace::kLog);

    naive_bayes::ClassDefinition classA;
    classA.name = "A";
    classA.prior = 0.5;
    classA.feature_names = {"x"};
    classA.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(0.0, 1.0));

    naive_bayes::ClassDefinition classB;
    classB.name = "B";
    classB.prior = 0.5;
    classB.feature_names = {"x"};
    classB.feature_models.push_back(std::make_unique<naive_bayes::Gaussian>(5.0, 1.0));

    clf.AddClassDefinition(std::move(classA));
    clf.AddClassDefinition(std::move(classB));
    clf.SetClassGroups({{"Left", {"A"}}, {"Right", {"B"}}});

    const std::vector<double> features = {0.0};
    const naive_bayes::pipeline::DetailedPrediction detailed =
        naive_bayes::pipeline::RunDetailedPrediction(clf, features);
    if (detailed.predicted_class != "A") {
        throw std::runtime_error("DetailedPrediction helper returned wrong class");
    }
    if (detailed.probabilities.size() != 2) {
        throw std::runtime_error("DetailedPrediction helper missing class posteriors");
    }
    if (detailed.group_probabilities.size() != 2) {
        throw std::runtime_error("DetailedPrediction helper missing grouped posteriors");
    }
    if (detailed.predicted_group.empty()) {
        throw std::runtime_error("DetailedPrediction helper missing predicted group");
    }

    const std::vector<std::pair<std::string, double>> zipped =
        naive_bayes::pipeline::BuildFeatureInputs({"x"}, features);
    if (zipped.size() != 1 || zipped[0].first != "x" || zipped[0].second != 0.0) {
        throw std::runtime_error("BuildFeatureInputs returned unexpected content");
    }

    bool threw = false;
    try {
        (void)naive_bayes::pipeline::BuildFeatureInputs({"x", "y"}, features);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("BuildFeatureInputs should throw on name/value mismatch");
    }
}

void TestClassModelIntrospection() {
    std::cout << "[Test] NaiveBayes class model introspection..." << std::endl;

    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_class_model_introspection.json";
    std::ofstream out(tmp);
    out << R"({
      "classes": [
        {
          "name": "ClassA",
          "prior": 0.6,
          "features": [
            { "name": "rcs", "type": "gaussian", "params": { "mean": 2.0, "sigma": 0.5 } },
            { "name": "length", "type": "gaussian", "params": { "mean": 3.0, "sigma": 0.4 } }
          ]
        },
        {
          "name": "ClassB",
          "prior": 0.4,
          "features": [
            { "name": "rcs", "type": "gaussian", "params": { "mean": 6.0, "sigma": 1.2 } },
            { "name": "length", "type": "gaussian", "params": { "mean": 7.0, "sigma": 0.9 } }
          ]
        }
      ]
    })";
    out.close();

    const naive_bayes::NaiveBayes model = naive_bayes::io::LoadModelConfiguration(tmp);
    const std::vector<naive_bayes::ClassModelInfo> classes = model.ClassModels();
    if (classes.size() != 2) {
        throw std::runtime_error("Expected 2 class models from introspection");
    }
    if (classes[0].features.size() != 2 || classes[1].features.size() != 2) {
        throw std::runtime_error("Expected 2 features per class in introspection");
    }
    if (classes[0].features[0].distribution_type != "gaussian") {
        throw std::runtime_error("Expected gaussian distribution type in introspection");
    }
    if (classes[0].features[0].params.size() != 2) {
        throw std::runtime_error("Expected gaussian params [mean,sigma]");
    }
}

void TestSampleFeatureForClassSupportsRayleigh() {
    std::cout << "[Test] Sampling from class feature supports rayleigh..." << std::endl;

    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "nb_sample_rayleigh_test.json";
    std::ofstream out(tmp);
    out << R"({
      "classes": [
        {
          "name": "ClassA",
          "prior": 1.0,
          "features": [
            { "name": "rcs", "type": "gaussian", "params": { "mean": 2.0, "sigma": 0.5 } },
            { "name": "length", "type": "rayleigh", "params": { "sigma": 3.0 } }
          ]
        }
      ]
    })";
    out.close();

    const naive_bayes::NaiveBayes model = naive_bayes::io::LoadModelConfiguration(tmp);
    std::mt19937 rng(7);
    const double sampled_length = model.SampleFeatureForClass(0, "length", rng);
    if (!(sampled_length > 0.0)) {
        throw std::runtime_error("Expected positive rayleigh sample for length");
    }
}

} // namespace

int RunTestSuite() {
    std::cout << "--- Running Naive Bayes Internal Test Suite ---\n";
    int failures = 0;
    
    try {
        TestDistributions();
    } catch (const std::exception& ex) {
        std::cerr << "!!! DISTRIBUTION TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestLogSumExpLogic();
    } catch (const std::exception& ex) {
        std::cerr << "!!! LOG-SUM-EXP TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestLayoutMapping();
    } catch (const std::exception& ex) {
        std::cerr << "!!! LAYOUT MAPPING TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestLiveJsonPredictionLoop();
    } catch (const std::exception& ex) {
        std::cerr << "!!! LIVE JSON HELPER TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestMissingFeatureInference();
    } catch (const std::exception& ex) {
        std::cerr << "!!! MISSING FEATURE INFERENCE TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestFeatureUnionAndMissingSlots();
    } catch (const std::exception& ex) {
        std::cerr << "!!! FEATURE UNION TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestDuplicateFeaturesRejected();
    } catch (const std::exception& ex) {
        std::cerr << "!!! DUPLICATE FEATURE TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestClassGroupPosteriors();
    } catch (const std::exception& ex) {
        std::cerr << "!!! GROUP POSTERIOR TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestCsvGroupColumns();
    } catch (const std::exception& ex) {
        std::cerr << "!!! GROUP CSV OUTPUT TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestCsvFeatureInputsAndIdColumns();
    } catch (const std::exception& ex) {
        std::cerr << "!!! FEATURE/ID CSV OUTPUT TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestDetailedPredictionHelperAndFeatureZip();
    } catch (const std::exception& ex) {
        std::cerr << "!!! SHARED HELPER TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestClassModelIntrospection();
    } catch (const std::exception& ex) {
        std::cerr << "!!! CLASS MODEL INTROSPECTION TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    try {
        TestSampleFeatureForClassSupportsRayleigh();
    } catch (const std::exception& ex) {
        std::cerr << "!!! FEATURE SAMPLING TEST FAILED: " << ex.what() << "\n";
        failures++;
    }

    if (failures == 0) {
        std::cout << "--- All tests passed successfully. ---\n";
        return 0;
    } else {
        std::cerr << "--- " << failures << " TEST(S) FAILED. ---\n";
        return 1;
    }
}

} // namespace naive_bayes::test
