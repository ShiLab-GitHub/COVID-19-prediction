# COVID-19-prediction

COVID-19 prediction is a machine learning model used to COVID-19. The model distinguishes samples from the five cohorts of critical COVID-19 recovered in ICU, non-critical COVID-19, non-COVID-19 in ICU, non-COVID-19 with symptoms and non-COVID-19 healthy by plasma metabolome data.

##Data source
The metabolomics raw data is from MetaboLights（ https://www.ebi.ac.uk/metabolights/ ）, Identifier: MTBLS1866

MTBLS1866:Large-Scale Plasma Analysis Revealed New Mechanisms and Molecules Associated with the Host Response to SARS-CoV-2.

category 0  critical COVID-19 recovered in ICU：19
category 1  non-critical COVID-19：84
category 2  non-COVID-19 in ICU：12
category 3  non-COVID-19 with symptoms：16
category 4  non-COVID-19 healthy：26


##Model


Random forest is a classifier that contains multiple decision trees, and its output category is determined by the mode of the category output by an individual tree.

We use Random forest model for COVID-19 metabolome data to achieve five classifications and  obtain the feature importance ranking. In addition, we give some evaluation metrics for machine learning.

We can get the importance ranking of features, the accuracy of classification, recall, precision, f1-score.
## The environment of DeepMCL-DTI
    Python 3.7.12
    Scikit 1.0.2
    Numpy 1.21.6
    Pandas 1.6.5

##Example output:

Feature sorting results ----------------------
1、(0.0211, 'CHEBI:89633')
2、(0.0136, 'CHEBI:35584')
3、(0.0094, 'CHEBI:48937')
4、(0.0062, 'CHEBI:183272')
5、(0.005, 'CHEBI:45993')
6、(0.0048, 'CHEBI:183278')
7、(0.0045, 'CHEBI:183293')
8、(0.0043, 'CHEBI:26984')
9、(0.0037, 'CHEBI:176811')
10、(0.0034, 'CHEBI:17368')
11、(0.0033, 'CHEBI:32938')
12、(0.0033, 'CHEBI:183270')
13、(0.0032, 'CHEBI:183260')
14、(0.0031, 'CHEBI:22470')
15、(0.003, 'CHEBI:15440')
16、(0.0029, 'CHEBI:183266')
17、(0.0027, 'CHEBI:183303')
18、(0.0026, 'CHEBI:75455')
19、(0.0026, 'CHEBI:422')
20、(0.0025, 'CHEBI:75456')
21、(0.0024, 'CHEBI:183304')
22、(0.0024, 'CHEBI:183276')
23、(0.0024, 'CHEBI:15586')
24、(0.0023, 'CHEBI:30813')
25、(0.0023, 'CHEBI:28875')
26、(0.0023, 'CHEBI:183307')
27、(0.0022, 'CHEBI:183292')
28、(0.0022, 'CHEBI:183282')
29、(0.0022, 'CHEBI:142245')
30、(0.0021, 'CHEBI:37621')
31、(0.0021, 'CHEBI:16113')
32、(0.002, 'CHEBI:79053')
33、(0.002, 'CHEBI:30805')
34、(0.0019, 'CHEBI:77513')
35、(0.0019, 'CHEBI:35998')
36、(0.0019, 'CHEBI:28837')
37、(0.0019, 'CHEBI:27732')
38、(0.0018, 'CHEBI:17987')
39、(0.0017, 'CHEBI:84223')
40、(0.0017, 'CHEBI:45919')
 ![image](https://user-images.githubusercontent.com/102600946/227543117-4879c99a-02da-4db1-943d-74eae19f9da7.png)

accuracy: 0.79
precision: 0 : 0.83
precision: 1 : 0.74
precision: 2 : 1.00
precision: 3 : 0.00
precision: 4 : 1.00
recall: 0 : 0.83
recall: 1 : 0.96
recall: 2 : 0.67
recall: 3 : 0.00
recall: 4 : 0.75
f1-score: 0 : 0.83
f1-score: 1 : 0.83
f1-score: 2 : 0.80
f1-score: 3 : 0.00
f1-score: 4 : 0.86
