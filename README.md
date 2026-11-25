# Data Science: Data-Driven Decision Making

[English](#english) | [Dansk](#dansk)

## English

Exam project from the course *Data Science: Data-Driven Decision Making* at Copenhagen Business School (2026), written in R. It compares supervised classification models and uses clustering to find structure in unlabelled data.

### What the project does

**Supervised learning (yeast data)**
- Logistic regression with stepwise variable selection
- Support vector machines with linear, polynomial and RBF kernels, tuned with cross-validation
- Classification trees and random forests (tuning of `mtry` using out-of-bag accuracy)
- k-nearest neighbours with leave-one-out cross-validation to choose k
- Comparison of all models on a separate hold-out dataset against a baseline
- Interpretability of the chosen SVM: permutation importance, a surrogate tree and Platt scaling for calibrated probabilities

**Unsupervised learning (cancer data)**
- Min-max normalisation of the data
- Hierarchical clustering with different distance measures and linkage methods
- k-means clustering, with the number of clusters chosen using the elbow method
- Comparison of the clusters from the two methods

### Files

| File | Content |
| --- | --- |
| `DS2026_Exam.R` | All code for the assignment |
| `yeastf.txt`, `yeastfadditional.txt` | Training data and hold-out data for classification |
| `cancerf.txt`, `heartf.txt` | Data for the unsupervised part |
| `appendix_output.txt` | Output from the script |

### How to run

Install the packages and run the script from the repository folder:

```r
install.packages(c("e1071", "tree", "randomForest", "FNN", "cluster", "scatterplot3d"))
source("DS2026_Exam.R")
```

**Tools:** R, e1071, randomForest, tree, FNN, cluster

---

## Dansk

Eksamensprojekt fra faget *Data Science: Data-Driven Decision Making* på Copenhagen Business School (2026), skrevet i R. Projektet sammenligner modeller til klassifikation og bruger klyngeanalyse til at finde struktur i data uden labels.

### Hvad projektet gør

**Superviseret læring (yeast-data)**
- Logistisk regression med trinvis variabeludvælgelse
- Support vector machines med lineær, polynomiel og RBF-kerne, tunet med krydsvalidering
- Klassifikationstræer og random forests (tuning af `mtry` ud fra out-of-bag-præcision)
- k-nærmeste naboer med leave-one-out-krydsvalidering til valg af k
- Sammenligning af alle modeller på et separat testdatasæt og mod en baseline
- Fortolkning af den valgte SVM: permutationsvigtighed, et surrogattræ og Platt-skalering til kalibrerede sandsynligheder

**Ikke-superviseret læring (cancer-data)**
- Min-max-normalisering af data
- Hierarkisk klyngeanalyse med forskellige afstandsmål og linkage-metoder
- k-means med valg af antal klynger ud fra albuemetoden
- Sammenligning af klyngerne fra de to metoder

### Filer

| Fil | Indhold |
| --- | --- |
| `DS2026_Exam.R` | Al kode til opgaven |
| `yeastf.txt`, `yeastfadditional.txt` | Trænings- og testdata til klassifikation |
| `cancerf.txt`, `heartf.txt` | Data til den ikke-superviserede del |
| `appendix_output.txt` | Output fra scriptet |

### Sådan køres koden

Installér pakkerne og kør scriptet fra mappen med repoet:

```r
install.packages(c("e1071", "tree", "randomForest", "FNN", "cluster", "scatterplot3d"))
source("DS2026_Exam.R")
```

**Værktøjer:** R, e1071, randomForest, tree, FNN, cluster
