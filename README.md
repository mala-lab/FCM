# Foreseeing Abnormality: Time Series Anomaly Prediction via Future Context Modeling
Identifying anomalies from time series data plays an important role in various fields such as infrastructure security, intelligent operation and maintenance, and space exploration.
Current research focuses on detecting the anomalies after they occur, which can lead to significant financial/reputation loss or infrastructure damage.
In this work we instead study a more practical yet very challenging problem, time series anomaly prediction, aiming at providing early warnings for abnormal events before their occurrence. 
To tackle this problem, we introduce a novel principled approach, namely **f**uture **c**ontext **m**odeling (**FCM**). Its key insight is that the future abnormal events in a target window can be accurately predicted if their preceding observation window exhibits any **subtle** difference to normal data. To effectively capture such differences, FCM first leverages long-term forecasting models to generate a discriminative future context based on the observation data, aiming to amplify those subtle but unusual difference. It then models a normality correlation of the observation data with the forecasting future context to complement the normality modeling of the observation data in foreseeing possible abnormality in the target window. A joint variate-time attention learning is also introduced in FCM to leverage both temporal signals and features of the time series data for more discriminative normality modeling in the aforementioned two views.
Comprehensive experiments on five datasets demonstrate that FCM gains good recall rate (70\%+) on multiple datasets and significantly outperforms all baselines in $F_{1}$ score. 
![Description of Image](https://github.com/mala-lab/FCM/blob/main/flow.png)

## Get Started
1. Install Python 3.9.13, PyTorch 1.11.0.
2. Download data. You can obtain two benchmarks from Google Cloud. The datasets are well pre-processed. For the SWaT dataset, you can apply for it by following its official tutorial. We unify the SWaT dataset to minute granularity and retain only continuous metrics:
3. Train and evaluate. You can reproduce the experiment results as follows:
```bash
bash ./script/run.sh
```


## Main Result



![Uploading image.png…]()
