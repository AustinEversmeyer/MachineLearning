# Ant Bayes Live

Small Ant-driven C++ demo that links against the BayesClassifier library and uses the
"live" in-memory inference helper.

## Build

```bash
ant build
```

## Run

```bash
ant run
```

## Notes

- The BayesClassifier project is treated as a third-party dependency located at
  `../BayesClassifier`.
- Model configuration files are staged into `app/build/config` during the build.
- The demo uses `pipeline::LoadModel` + `PredictResultFromJsonObject`.
