cmake_minimum_required(VERSION 3.16)
project(naive_bayes_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(naive_bayes_core
  src/core/gaussian.cpp
  src/core/rayleigh.cpp
  src/core/factory.cpp
  src/core/naive_bayes.cpp
  src/core/prediction_helpers.cpp
  src/io/model_loader.cpp
  src/io/output_write.cpp
)
target_include_directories(naive_bayes_core PUBLIC include)

add_executable(naive_bayes_cli
  src/pipeline/main.cpp
  src/pipeline/pipeline_helpers.cpp
  src/pipeline/config_resolve.cpp
  src/pipeline/config_parse.cpp
  src/io/input_load.cpp
  src/test/tester.cpp
)
target_link_libraries(naive_bayes_cli PRIVATE naive_bayes_core)
