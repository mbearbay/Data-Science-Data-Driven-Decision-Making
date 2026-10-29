##################################################################
#  Data Science: Data Driven Decision Making - Final Assignment 2026
##################################################################

# Structure #####
rm(list = ls())
if (!is.null(dev.list())) dev.off()

#install.packages
# install.packages("e1071", dependencies = TRUE)
# install.packages("tree", dependencies = TRUE)
# install.packages("randomForest", dependencies = TRUE)
# install.packages("FNN", dependencies = TRUE)
# install.packages("cluster")
# install.packages("scatterplot3d")

#library
library(e1071)
library(tree)
library(randomForest)
library(FNN)
library(cluster)
library(scatterplot3d)

# Data ####
setwd("/Users/..../Data Driven Decision Making/Exam")
yeast           <- read.table("yeastf.txt",           header = TRUE)
yeastadditional <- read.table("yeastfadditional.txt", header = TRUE)
cancer          <- read.table("cancerf.txt",          header = TRUE)
heart           <- read.table("heartf.txt",           header = TRUE)
dir.create("R code 2026", showWarnings = FALSE)

# Supervised Learning ############################################
## Data preparation ----
dim(yeast); dim(yeastadditional)
str(yeast)
any(is.na(yeast)); any(is.na(yeastadditional))

yeast$class           <- as.factor(yeast$class)
yeastadditional$class <- as.factor(yeastadditional$class)
table(yeast$class) / nrow(yeast)
table(yeastadditional$class) / nrow(yeastadditional)

yeastNORM           <- yeast
yeastadditionalNORM <- yeastadditional
for (i in 1:(ncol(yeast) - 1)) {
  minimum <- min(yeast[, i])
  maximum <- max(yeast[, i])
  yeastNORM[, i]           <- as.vector(scale(yeast[, i],           center = minimum, scale = maximum - minimum))
  yeastadditionalNORM[, i] <- as.vector(scale(yeastadditional[, i], center = minimum, scale = maximum - minimum))
}

accuracy <- function(tab) sum(diag(tab)) / sum(tab)
metrics <- function(pred, actual) {
  tab  <- table(pred = factor(pred, levels = c("0", "1")), actual = actual)
  sens <- tab["1", "1"] / sum(tab[, "1"])
  spec <- tab["0", "0"] / sum(tab[, "0"])
  c(Accuracy = sum(diag(tab)) / sum(tab), Sensitivity = sens,
    Specificity = spec, BalancedAcc = (sens + spec) / 2)
}

## 1(i) Logistic Regression ----
LRmodel <- glm(class ~ ., family = "binomial", data = yeastNORM)
summary(LRmodel)
LRmodelR <- step(LRmodel, trace = 1)
summary(LRmodelR)
table(f6 = yeast$f6, class = yeast$class)

probabilitiesLR_train <- predict(LRmodelR, yeastNORM, type = "response")
predictionLR_train    <- ifelse(probabilitiesLR_train > 0.5, "1", "0")
accLR_train <- accuracy(table(pred = predictionLR_train, actual = yeastNORM$class))

set.seed(2026)
folds  <- sample(rep(1:10, length.out = nrow(yeastNORM)))
cvpred <- rep(NA, nrow(yeastNORM))
for (f in 1:10) {
  fit <- glm(formula(LRmodelR), family = "binomial", data = yeastNORM[folds != f, ])
  cvpred[folds == f] <- ifelse(predict(fit, yeastNORM[folds == f, ], type = "response") > 0.5, "1", "0")
}
accLR_cv <- mean(cvpred == yeastNORM$class)

probabilitiesLR <- predict(LRmodelR, yeastadditionalNORM, type = "response")
predictionLR    <- ifelse(probabilitiesLR > 0.5, "1", "0")
table(pred = predictionLR, actual = yeastadditionalNORM$class)
accLR_train; accLR_cv

## 1(ii) Support Vector Machines ----
### Linear kernel ----
set.seed(2026)
tunedSVMlinear <- tune.svm(class ~ ., data = yeastNORM,
                           cost = 2^(-12:12), kernel = "linear")
summary(tunedSVMlinear)
bestcost_lin <- tunedSVMlinear$best.parameters$cost
SVMlinear <- svm(class ~ ., data = yeastNORM, cost = bestcost_lin, kernel = "linear")
predSVMlinear <- predict(SVMlinear, yeastadditionalNORM[, -9])
classificationtableSVMlinear <- table(pred = predSVMlinear, actual = yeastadditionalNORM$class)
classificationtableSVMlinear

### Polynomial kernel ----
set.seed(2026)
tunedSVMpoly <- tune.svm(class ~ ., data = yeastNORM, kernel = "polynomial",
                         cost = 2^(-6:6), gamma = 2^(-3:0), degree = 2:4, coef0 = 0:1)
bestpoly <- tunedSVMpoly$best.parameters
bestpoly
SVMpoly <- svm(class ~ ., data = yeastNORM, kernel = "polynomial",
               cost = bestpoly$cost, gamma = bestpoly$gamma,
               degree = bestpoly$degree, coef0 = bestpoly$coef0)
predSVMpoly <- predict(SVMpoly, yeastadditionalNORM[, -9])
classificationtableSVMpoly <- table(pred = predSVMpoly, actual = yeastadditionalNORM$class)

### RBF kernel ----
set.seed(2026)
tunedSVMRBF <- tune.svm(class ~ ., data = yeastNORM,
                        cost = 2^(-12:12), gamma = 2^(-12:12))
bestgamma_rbf <- tunedSVMRBF$best.parameters$gamma
bestcost_rbf  <- tunedSVMRBF$best.parameters$cost
SVMRBF <- svm(class ~ ., data = yeastNORM, cost = bestcost_rbf, gamma = bestgamma_rbf)
predSVMRBF <- predict(SVMRBF, yeastadditionalNORM[, -9])
classificationtableSVMRBF <- table(pred = predSVMRBF, actual = yeastadditionalNORM$class)

### Summary SVM kernels ----
SVMsummary <- data.frame(
  Kernel     = c("Linear", "Polynomial", "RBF"),
  Parameters = c(paste0("C=", bestcost_lin),
                 paste0("C=", bestpoly$cost, ", gamma=", bestpoly$gamma,
                        ", d=", bestpoly$degree, ", coef0=", bestpoly$coef0),
                 paste0("C=", bestcost_rbf, ", gamma=", bestgamma_rbf)),
  CVaccuracy = 1 - c(tunedSVMlinear$best.performance, tunedSVMpoly$best.performance,
                     tunedSVMRBF$best.performance),
  Train      = c(mean(predict(SVMlinear) == yeastNORM$class),
                 mean(predict(SVMpoly)   == yeastNORM$class),
                 mean(predict(SVMRBF)    == yeastNORM$class)),
  Additional = c(accuracy(classificationtableSVMlinear), accuracy(classificationtableSVMpoly),
                 accuracy(classificationtableSVMRBF)),
  nSV        = c(SVMlinear$tot.nSV, SVMpoly$tot.nSV, SVMRBF$tot.nSV))
SVMsummary

bestSVMrow <- which.max(SVMsummary$CVaccuracy)
SVMfinal   <- list(SVMlinear, SVMpoly, SVMRBF)[[bestSVMrow]]
predSVM    <- list(predSVMlinear, predSVMpoly, predSVMRBF)[[bestSVMrow]]
cat("Chosen SVM kernel:", SVMsummary$Kernel[bestSVMrow], "\n")

## 1(iii) Classification Trees ----
mytree <- tree(class ~ ., data = yeast)
summary(mytree)
plot(mytree); text(mytree, cex = 0.7)
title("Full classification tree (yeastf)")

set.seed(2026)
mycrossval <- cv.tree(mytree, FUN = prune.tree, K = 10)
devbestsize <- mycrossval$size[which.min(mycrossval$dev)]
devbestsize
table(predict(prune.tree(mytree, best = devbestsize), yeast, type = "class"))

set.seed(2026)
mycrossvalmis <- cv.tree(mytree, FUN = prune.misclass, K = 10)
mybestsize   <- mycrossvalmis$size[which(mycrossvalmis$dev == min(mycrossvalmis$dev))]
myprunedtree <- prune.misclass(mytree, best = mybestsize[1])

predTree <- predict(myprunedtree, yeastadditional[, -9], type = "class")
table(pred = predTree, actual = yeastadditional$class)

cvpredtree <- rep(NA, nrow(yeast))
for (f in 1:10) {
  fit <- tree(class ~ ., data = yeast[folds != f, ])
  if (mybestsize[1] < summary(fit)$size) fit <- prune.misclass(fit, best = mybestsize[1])
  cvpredtree[folds == f] <- as.character(predict(fit, yeast[folds == f, ], type = "class"))
}
accTree_cv    <- mean(cvpredtree == yeast$class)
accTree_train <- mean(predict(myprunedtree, yeast, type = "class") == yeast$class)
accTree_train; accTree_cv

## 1(iv) Random Forests ----
oobmtry <- NULL
for (m in 1:8) {
  set.seed(2026)
  rfm <- randomForest(class ~ ., data = yeast, ntree = 500, mtry = m)
  oobmtry[m] <- rfm$err.rate[500, "OOB"]
}
plot(1:8, 1 - oobmtry, type = "b", pch = 16, xlab = "mtry", ylab = "OOB accuracy",
     main = "Random forest: tuning mtry (500 trees)")
bestmtry <- which.min(oobmtry)

set.seed(2026)
myrfbest <- randomForest(class ~ ., data = yeast, ntree = 500, mtry = bestmtry, importance = TRUE)
print(myrfbest)
importance(myrfbest)
varImpPlot(myrfbest, main = "")
title("Random forest: Variable importance", line = -1)

predRF <- predict(myrfbest, yeastadditional[, -9], type = "class")
table(pred = predRF, actual = yeastadditional$class)
accRF_train <- mean(predict(myrfbest, yeast) == yeast$class)
accRF_oob   <- 1 - myrfbest$err.rate[500, "OOB"]

## 1(v) k-Nearest Neighbors ----
myxtrainNORM <- yeastNORM[, -9]
myytrainNORM <- yeastNORM[, 9]
myxtestNORM  <- yeastadditionalNORM[, -9]
myytestNORM  <- yeastadditionalNORM[, 9]

bestk <- 0; bestaccuracy <- 0; accuracyk <- NULL
for (auxk in 1:30) {
  set.seed(2026)
  mycv <- knn.cv(train = myxtrainNORM, cl = myytrainNORM, k = auxk)
  mytable <- table(mycv, myytrainNORM)
  accuracyk[auxk] <- sum(diag(mytable)) / sum(mytable)
  if (bestaccuracy < accuracyk[auxk]) bestk <- auxk
  if (bestaccuracy < accuracyk[auxk]) bestaccuracy <- accuracyk[auxk]
}
plot(accuracyk, type = "b", xlab = "K", ylab = "LOOCV accuracy", pch = 16,
     main = "k-NN: leave-one-out cross-validation")
abline(v = bestk, lty = 2)
bestk; bestaccuracy
which.max(accuracyk[1:15]); max(accuracyk[1:15])

set.seed(2026)
mybestknn <- knn(train = myxtrainNORM, test = myxtestNORM, cl = myytrainNORM, k = bestk)
table(pred = mybestknn, actual = myytestNORM)
set.seed(2026)
accKNN_train <- mean(knn(myxtrainNORM, myxtrainNORM, myytrainNORM, k = bestk) == myytrainNORM)

## 1(vi) Model Comparison on yeastfadditional ----
baseline <- mean(yeastadditional$class == "0")
baseline

allpredictions <- list(LR = predictionLR, `SVM linear` = predSVMlinear,
                       `SVM poly` = predSVMpoly, `SVM RBF` = predSVMRBF,
                       Tree = predTree, RF = predRF, kNN = mybestknn)

comparison <- data.frame(
  Model  = c("Logistic Regression", "SVM linear", "SVM polynomial", "SVM RBF",
             "Classification Tree", "Random Forest", "k-NN"),
  CVtype = c("10-fold CV", "10-fold CV", "10-fold CV", "10-fold CV", "10-fold CV",
             "OOB", "Leave-one-out CV"),
  Train  = c(accLR_train, SVMsummary$Train, accTree_train, accRF_train, accKNN_train),
  CV     = c(accLR_cv, SVMsummary$CVaccuracy, accTree_cv, accRF_oob, bestaccuracy))

counts <- t(sapply(allpredictions, function(p) {
  tab <- table(pred = factor(as.character(p), levels = c("0", "1")), actual = yeastadditional$class)
  c(TN = tab["0", "0"], FP = tab["1", "0"], FN = tab["0", "1"], TP = tab["1", "1"])
}))
detailed <- t(sapply(allpredictions, metrics, actual = yeastadditional$class))
comparison <- cbind(comparison, counts, detailed)
comparison$BeatsBaseline <- ifelse(comparison$Accuracy > baseline + 1e-9, "Yes", "No")
comparison$Rank <- rank(-round(comparison$Accuracy, 6), ties.method = "min")
rownames(comparison) <- NULL
comparison

## 1(viii) SVM Interpretability and Trustworthiness ----
### Permutation importance ----
set.seed(2026)
baseacc <- mean(predict(SVMfinal, yeastadditionalNORM[, -9]) == yeastadditionalNORM$class)
permimp <- sapply(1:8, function(j) {
  mean(replicate(30, {
    Xp <- yeastadditionalNORM[, -9]
    Xp[, j] <- sample(Xp[, j])
    baseacc - mean(predict(SVMfinal, Xp) == yeastadditionalNORM$class)
  }))
})
names(permimp) <- colnames(yeastNORM)[1:8]
round(sort(permimp, decreasing = TRUE), 4)

### Surrogate tree ----
surrogatedata <- yeast[, -9]
surrogatedata$svmclass <- predict(SVMfinal, yeastNORM[, -9])
surrogate <- tree(svmclass ~ ., data = surrogatedata)
set.seed(2026)
cvsur <- cv.tree(surrogate, FUN = prune.misclass)
surrogate <- prune.misclass(surrogate, best = max(2, cvsur$size[which.min(cvsur$dev)]))
mean(predict(surrogate, yeastadditional[, -9], type = "class") ==
       predict(SVMfinal, yeastadditionalNORM[, -9]))

### Platt scaling ----
set.seed(2026)
SVMprob <- svm(class ~ ., data = yeastNORM, cost = bestcost_rbf, gamma = bestgamma_rbf,
               probability = TRUE)
probSVM <- attr(predict(SVMprob, yeastadditionalNORM[, -9], probability = TRUE),
                "probabilities")[, "1"]
confident <- abs(probSVM - 0.5) > 0.25
tapply(ifelse(probSVM > 0.5, "1", "0") == yeastadditionalNORM$class, confident, mean)
table(confident)

# Unsupervised Learning ##########################################
## Data preparation ----
dim(cancer)
any(is.na(cancer))
minimum <- apply(cancer, 2, min)
maximum <- apply(cancer, 2, max)
cancerNORM <- as.data.frame(scale(cancer, center = minimum, scale = maximum - minimum))

adjrand <- function(a, b) {
  tab <- table(a, b); n <- sum(tab)
  sc  <- sum(choose(tab, 2)); sa <- sum(choose(rowSums(tab), 2)); sb <- sum(choose(colSums(tab), 2))
  expct <- sa * sb / choose(n, 2)
  (sc - expct) / ((sa + sb) / 2 - expct)
}

## 2(i) Hierarchical Clustering ----
distances <- c("euclidean", "manhattan", "maximum")
linkages  <- c("complete", "single", "average", "ward.D2")
hclusts   <- list()
for (d in distances) for (l in linkages) {
  hclusts[[paste(d, l)]] <- hclust(dist(cancerNORM, method = d), method = l)
}

par(mfrow = c(3, 4))
for (nm in names(hclusts)) plot(hclusts[[nm]], hang = -1, cex = 0.4, main = nm, xlab = "", sub = "")
par(mfrow = c(1, 1))

hcselection <- do.call(rbind, lapply(names(hclusts), function(nm) {
  d  <- strsplit(nm, " ")[[1]][1]
  D  <- dist(cancerNORM, method = d)
  cl <- cutree(hclusts[[nm]], k = 2)
  data.frame(Combination   = nm,
             Sizes_k2      = paste(sort(table(cl), decreasing = TRUE), collapse = "/"),
             Smallest      = min(table(cl)),
             Silhouette_k2 = round(mean(silhouette(cl, D)[, 3]), 3))
}))
hcselection$Valid <- hcselection$Smallest >= 0.10 * nrow(cancerNORM)
hcselection

valid     <- hcselection[hcselection$Valid, ]
bestcombo <- valid$Combination[which.max(valid$Silhouette_k2)]
bestcombo

hcBest <- hclusts[[bestcombo]]
plot(hcBest, hang = -1, cex = 0.6, main = paste("cancerf:", bestcombo, "(chosen)"),
     xlab = "Tumor", sub = "")
rect.hclust(hcBest, k = 2, border = "red")
rect.hclust(hcBest, k = 3, border = "blue")

rev(hcBest$height)[1:5]
labelsH2 <- cutree(hcBest, k = 2)
labelsH3 <- cutree(hcBest, k = 3)
table(labelsH2); table(labelsH3)
Dbest <- dist(cancerNORM, method = strsplit(bestcombo, " ")[[1]][1])
c(Silhouette_k2 = mean(silhouette(labelsH2, Dbest)[, 3]),
  Silhouette_k3 = mean(silhouette(labelsH3, Dbest)[, 3]))
round(t(aggregate(cancer[, c(1, 4, 7)], list(cluster = labelsH2), mean)), 3)

## 2(ii) Partitioning-Based Clustering ----
tot_withinss <- NULL
for (auxk in 1:15) {
  set.seed(2026)
  tot_withinss[auxk] <- kmeans(cancerNORM, auxk, nstart = 50)$tot.withinss
}
plot(1:15, tot_withinss, type = "b", pch = 16, xlab = "k",
     ylab = "Total within-cluster sum of squares", main = "k-means: elbow plot")
round(tot_withinss, 1)
round(1 - tot_withinss / tot_withinss[1], 3)

kchosen <- 2
set.seed(2026)
Kmeansfinal <- kmeans(cancerNORM, kchosen, nstart = 50)
Kmeansfinal$size
round(t(aggregate(cancer[, c(1, 4, 7)], list(cluster = Kmeansfinal$cluster), mean)), 3)
pairs(cancerNORM[, 1:10], col = c("red", "blue")[Kmeansfinal$cluster],
      pch = 16, cex = 0.5, main = "k-means clusters (mean variables)")
table(HC = labelsH2, Kmeans = Kmeansfinal$cluster)

## 2(iv) Robust Clustering Against Outliers ----
set.seed(2026)
outrows  <- sample(1:nrow(cancer), 3)
outliers <- cancer[outrows, ]
outliers[, 21:30] <- outliers[, 21:30] * 4
cancerOUT <- rbind(cancer, outliers)
rownames(cancerOUT) <- c(1:100, paste0("OUT", 1:3))
minO <- apply(cancerOUT, 2, min); maxO <- apply(cancerOUT, 2, max)
cancerOUTNORM <- as.data.frame(scale(cancerOUT, center = minO, scale = maxO - minO))

set.seed(2026)
KmeansOUT <- kmeans(cancerOUTNORM, 2, nstart = 50)
ARI_kmeans_outliers <- adjrand(KmeansOUT$cluster[1:100], Kmeansfinal$cluster)
tabOUT <- table(original = Kmeansfinal$cluster, withoutliers = KmeansOUT$cluster[1:100])
tabOUT

tot_withinssOUT <- NULL
for (auxk in 1:15) {
  set.seed(2026)
  tot_withinssOUT[auxk] <- kmeans(cancerOUTNORM, auxk, nstart = 50)$tot.withinss
}
plot(1:15, tot_withinssOUT / tot_withinssOUT[1], type = "b", col = "red", pch = 16,
     xlab = "k", ylab = "Within SS / total SS", main = "Elbow plot with and without outliers")
lines(1:15, tot_withinss / tot_withinss[1], type = "b", col = "blue", pch = 16)
legend("topright", c("with 3 outliers", "original"), col = c("red", "blue"), lty = 1)

### k-medoids ----
cancerOUTROB <- as.data.frame(scale(cancerOUT, center = apply(cancerOUT, 2, median),
                                    scale = apply(cancerOUT, 2, mad)))
cancerROB    <- as.data.frame(scale(cancer, center = apply(cancer, 2, median),
                                    scale = apply(cancer, 2, mad)))
pamOUT  <- pam(cancerOUTROB, 2, metric = "manhattan")
pamORIG <- pam(cancerROB,    2, metric = "manhattan")
ARI_pam_outliers <- adjrand(pamOUT$clustering[1:100], pamORIG$clustering)

### Trimmed k-means ----
trimmed_kmeans <- function(X, k, alpha = 0.05, nstart = 20, iter = 50) {
  X <- as.matrix(X); n <- nrow(X); h <- floor(n * (1 - alpha)); best <- NULL
  for (s in 1:nstart) {
    centres <- X[sample(n, k), , drop = FALSE]
    for (it in 1:iter) {
      d2 <- sapply(1:k, function(j) rowSums((X - matrix(centres[j, ], n, ncol(X), byrow = TRUE))^2))
      cl <- apply(d2, 1, which.min); dmin <- d2[cbind(1:n, cl)]
      keep <- rank(dmin, ties.method = "first") <= h
      newc <- t(sapply(1:k, function(j) colMeans(X[keep & cl == j, , drop = FALSE])))
      if (any(is.na(newc)) || max(abs(newc - centres)) < 1e-8) break
      centres <- newc
    }
    obj <- sum(dmin[keep])
    if (is.null(best) || obj < best$obj) { cl[!keep] <- 0; best <- list(cluster = cl, obj = obj, centres = centres) }
  }
  best
}
set.seed(2026)
tkmOUT <- trimmed_kmeans(cancerOUTNORM, 2, alpha = 0.05)
tkmOUT$cluster[101:103]
keep100 <- tkmOUT$cluster[1:100] != 0
ARI_tkm_outliers <- adjrand(tkmOUT$cluster[1:100][keep100], Kmeansfinal$cluster[keep100])

### Outlier detection and removal ----
pcOUT   <- prcomp(cancerOUT, scale. = TRUE)$x[, 1:5]
mahal   <- mahalanobis(pcOUT, colMeans(pcOUT), cov(pcOUT))
flagged <- which(mahal > qchisq(0.99, df = 5))
rownames(cancerOUT)[flagged]
cleaned <- cancerOUT[-flagged, ]
minC <- apply(cleaned, 2, min); maxC <- apply(cleaned, 2, max)
set.seed(2026)
KmeansCLEAN <- kmeans(scale(cleaned, center = minC, scale = maxC - minC), 2, nstart = 50)
real <- as.numeric(rownames(cleaned)[rownames(cleaned) %in% as.character(1:100)])
ARI_clean <- adjrand(KmeansCLEAN$cluster[rownames(cleaned) %in% as.character(1:100)],
                     Kmeansfinal$cluster[real])

robusttable <- data.frame(
  Method = c("k-means (min-max) with outliers", "PAM, Manhattan, robust scaling",
             "Trimmed k-means (alpha = 5%)", "Remove outliers (Mahalanobis) + k-means"),
  ARI = round(c(ARI_kmeans_outliers, ARI_pam_outliers, ARI_tkm_outliers, ARI_clean), 3))
robusttable

# Visualization ##################################################
## Data preparation ----
dim(heart)
binaryvars     <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking")
continuousvars <- setdiff(colnames(heart), binaryvars)
heartSTD <- scale(heart, center = TRUE, scale = TRUE)

## 3(i) Principal Component Analysis ----
myPCA <- prcomp(heart, center = TRUE, scale. = TRUE)
summary(myPCA)
round(myPCA$rotation[, 1:2], 3)
sum(myPCA$sdev^2 > 1)

explained  <- myPCA$sdev^2 / sum(myPCA$sdev^2)
cumulative <- cumsum(explained)
par(mfrow = c(1, 2))
plot(explained, type = "b", pch = 16, xlab = "Principal component",
     ylab = "Proportion of variance explained", main = "Scree plot")
# abline(h = 1 / 12, lty = 2, col = "red")
plot(cumulative, type = "b", pch = 16, xlab = "Number of principal components",
     ylab = "Cumulative proportion", main = "Cumulative variance explained", ylim = c(0, 1))
# abline(h = 0.8, lty = 2, col = "red")
par(mfrow = c(1, 1))

x <- myPCA$x[, 1]; y <- myPCA$x[, 2]; z <- myPCA$x[, 3]
plot(x, y, xlab = paste0("PC1 (", round(100 * explained[1], 1), "%)"),
     ylab = paste0("PC2 (", round(100 * explained[2], 1), "%)"),
     main = "PCA of heartf", pch = 16, col = ifelse(heart$sex == 1, "steelblue", "tomato"))
legend("topright", c("male", "female"), col = c("steelblue", "tomato"), pch = 16, cex = 0.8)

scatterplot3d(x, y, z, xlab = "PC1", ylab = "PC2", zlab = "PC3", angle = 45,
              color = ifelse(heart$sex == 1, "steelblue", "tomato"), pch = 16,
              main = "PCA of heartf (3D)")
legend("topright", c("male", "female"), col = c("steelblue", "tomato"), pch = 16, cex = 0.8)

myPCAcont <- prcomp(heart[, continuousvars], center = TRUE, scale. = TRUE)
summary(myPCAcont)

## 3(ii) MultiDimensional Scaling ----
MDSeuc <- cmdscale(dist(heartSTD, method = "euclidean"), k = 2, eig = TRUE)
MDSman <- cmdscale(dist(heartSTD, method = "manhattan"), k = 2, eig = TRUE)
MDSmax <- cmdscale(dist(heartSTD, method = "maximum"),   k = 2, eig = TRUE)
Dgower <- daisy(heart, metric = "gower", type = list(asymm = binaryvars))
MDSgow <- cmdscale(Dgower, k = 2, eig = TRUE)

GOF <- rbind(Euclidean = MDSeuc$GOF, Manhattan = MDSman$GOF,
             Maximum = MDSmax$GOF, Gower = MDSgow$GOF)
colnames(GOF) <- c("GOF (abs eig)", "GOF (pos eig)")
round(GOF, 3)

par(mfrow = c(2, 2), mar = c(3, 3, 2, 1))
for (m in list(list(MDSeuc, "Euclidean"), list(MDSman, "Manhattan"),
               list(MDSmax, "Maximum"), list(MDSgow, "Gower (mixed data)"))) {
  plot(m[[1]]$points, xlab = "Coordinate 1", ylab = "Coordinate 2", pch = 16,
       col = ifelse(heart$sex == 1, "steelblue", "tomato"), main = paste("MDS -", m[[2]]))
}
par(mfrow = c(1, 1))

round(cor(MDSeuc$points[, 1], myPCA$x[, 1]), 4)
round(cor(MDSeuc$points[, 2], myPCA$x[, 2]), 4)
round(range(heartSTD[, "sex"]), 2)

heartNORM <- scale(heart, center = apply(heart, 2, min),
                   scale = apply(heart, 2, max) - apply(heart, 2, min))
round(cmdscale(dist(heartNORM, method = "euclidean"), k = 2, eig = TRUE)$GOF, 3)

# Output for report #############################################
sink("R code 2026/appendix_output.txt")
cat("== Logistic regression, full model\n");     print(summary(LRmodel))
cat("\n== Stepwise selection\n");                 print(LRmodelR$anova)
cat("\n== SVM tuning, best parameters\n");        print(SVMsummary)
cat("\n== Cluster means (hierarchical, k = 2)\n")
print(round(t(aggregate(cancer[, c(1, 4, 7)], list(cluster = labelsH2), mean)), 3))
cat("\n== PCA\n");                                 print(summary(myPCA)); print(round(myPCA$rotation[, 1:2], 3))
cat("\n== MDS goodness of fit\n");                 print(round(GOF, 3))
sink()
