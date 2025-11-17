
#!/bin/bash

for val in 0.1 0.05 0.01
do
  echo "Running with TA = ${val}"
  python adarank_roberta_glue.py --exp_config="config/roberta_TA_${val}.yaml"
done