data.path <- "/scratch/qg251/dbart_mid/Civil_War_PA"
script.path <- "/scratch/qg251/dbart_mid/muchlinski_rep"

options(java.parameters = "-Xmx300g")
data <- read.csv(file=paste0(data.path, "/SambnisImp.csv")) # data for prediction
data2 <- read.csv(file=paste0(data.path, "/Amelia.Imp3.csv")) # data for causal machanisms

# data$peaceyears <- 0
# dyad_list <- unique(data$cowcode)
# for (d in dyad_list){
#  index <- which(data$cowcode == d)
#  if(length(index) > 1){
#    i = 0
#    for (x in min(index[-1]):max(index)){
#      if(!((data$warstds[x] == 1 & data$warstds[x] == data$warstds[x-1]) | (data$warstds[x] == 0 & data$warstds[x-1] == 1))){
#        data$peaceyears[x] <- data$peaceyears[x-1] + 1
#      }
#      else{
#        data$peaceyears[x] <- 0
#      }
#    }
#  }
# }

library(foreign)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(randomForest) #for random forests
library(caret) # for CV folds and data splitting
library(ROCR) # for diagnostics and ROC plots/stats
library(pROC) # same as ROCR
#library(stepPlr) # Firth;s logit implemented thru caret library
# library(doMC) # for using multipe processor cores 
library(bartMachine)
###Using only the 88 variables specified in Sambanis (2006) Appendix###
data.full<-data[,c("warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
        "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
        "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
        "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
        "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
        "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
        "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
        "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
        "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
        "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
        "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
        "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
        "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
        "warhist", "xconst")]

###Converting DV into Factor with names for Caret Library###
data.full$warstds<-factor(
  data.full$warstds,
  levels=c(0,1),
  labels=c("peace", "war"))

# registerDoMC(cores=7) # distributing workload over multiple cores for faster computaiton

set.seed(666) #the most metal seed for CV

# data.full <- sample_frac(data.full, size = 0.2, replace = FALSE)

#This method of data slicing - or CV - will be used for all logit models - uncorrected and corrected
tc<-trainControl(method="cv",
                 number=10,#creates CV folds - 5 for this data
                 summaryFunction=twoClassSummary, # provides ROC summary stats in call to model
                 classProb=T)

#Fearon and Laitin Model Specification###
model.fl.1<-train(as.factor(warstds)~warhist+ln_gdpen+lpopns+lmtnest+ncontig+oil+nwstate
             +inst3+pol4+ef+relfrac, #FL 2003 model spec
             metric="ROC", method="glm", family="binomial", #uncorrected logistic model
             trControl=tc, data=data.full)

summary(model.fl.1) #provides coefficients & traditional R model output
model.fl.1 # provides CV summary stats # keep in mind caret takes first class (here that's 0)
#as refernce class so sens and spec are backwards

data$peaceyears

###Now doing Collier and Hoeffler (2004) uncorrected logistic specification###
model.ch.1<-train(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
                  +lpopns+coldwar+seceduc+ptime, #CH 2004 model spec
                  metric="ROC", method="glm", family="binomial",
                  trControl=tc, data=data.full)
model.ch.1

###Implementing RF (with CV) on entirety of data###

model.rf<-train(as.factor(warstds)~.,
                  metric="ROC", method="rf",
                  sampsize=c(30,90), #Downsampling the class-imbalanced DV
                    importance=T, # Variable importance measures retained
                   proximity=F, ntree=1000, # number of trees grown
                   trControl=tc, data=data.full)
model.rf

#confusionMatrix(model.rf, norm="average")


bartGrid <- expand.grid(num_trees = c(500, 1000), k = 2, alpha = 0.95, beta = 2, nu = 3)
model.bt <- train(as.factor(warstds)~., data=data.full, metric = "ROC", method = "bartMachine",
                  tuneGrid = bartGrid, trControl = tc,  num_burn_in = 2000, num_iterations_after_burn_in = 2000, serialize = T)


###Random Forests on Amelia Imputed Data for Variable Importance Plot###
###Data Imputed only for Theoretically Important Variables### Done to analyze causal mechanisms

# myvars <- names(data2) %in% c("X", "country", "year", "atwards")
# newdata <- data2[!myvars]
#
# RF.out.am<-randomForest(as.factor(warstds)~.,sampsize=c(30, 90),
#           importance=T, proximity=F, ntree=1000, confusion=T, err.rate=T, data=newdata)
#
# varImpPlot(RF.out.am, sort=T, type=2, main="Variables Contributing Most to Predictive Accuracy of
#  Random Forests",n.var=20)



###Creating dotplot for Variable Importance Plot for RF###
# importance(RF.out.am)
#
# x<-c(2.3246380, 2.3055470, 1.8544127, 1.7447636, 1.6848230, 1.6094923,
#      1.5564487, 1.4832437, 1.4100489, 1.3116247, 1.2875924, 1.1799487,
#      1.1034743, 1.0983414, 1.0689367, 1.0663479, 1.0123892, 0.9961138, 0.9961138, 0.9922545)
# g<-c("GDP Growth", "GDP per Capita",
#      "Life Expectancy", "Western Europe and US Dummy", "Infant Mortality",
#      "Trade as Percent of GDP", "Mountainous Terrain", "Illiteracy Rate",
#      "Population (logged)", "Linguistic Hetrogeneity", "Anocracy", "Median Regional Polity Score",
#      "Primary Commodity Exports (Squared)", "Democracy", "Military Power", "Population Density",
#      "Political Instability", "Ethnic Fractionalization", "Secondary Education",
#      "Primary Commodity Exports")
# dotchart(rev(x), rev(g), cex=0.75, xlab="Mean Decrease in Gini Score (OOB Estimates)",
#          main="Variable Importance for Random Forests",
#          xlim=c(1, 2.5))

###ROC Plots for Different Models###


###Gathering info for ROC Plots ###
FL.1.pred<-predict(model.fl.1, type="prob")
RF.1.pred<-predict(model.rf, type="prob")
BT.1.pred<-predict(model.bt, type="prob")
# for bartMachine, the predicted probabilities for war and peace are reversed so need to make corrections for this situation
names(BT.1.pred) <- names(BT.1.pred)[order(names(BT.1.pred), decreasing = T)]

produce_ROC_AUC <- function(pred){
  pred <- prediction(pred$war, data.full$warstds)
  perf <- performance(pred,"tpr","fpr")
  AUC <- round(performance(pred,"auc")@y.values[[1]], digits = 3)
  return(list(perf, AUC))
}

FL <- produce_ROC_AUC(FL.1.pred)
RF <- produce_ROC_AUC(RF.1.pred)
BT <- produce_ROC_AUC(BT.1.pred)

pdf(paste0(script.path, "/roc_compare_10_fold.pdf"), height = 8, width = 8)
plot(FL[[1]], main="Logits, Random Forests and BART")
plot(RF[[1]], add=T, lty=2)
plot(BT[[1]], add=T, lty=3)
legend(0.32, 0.25, c(paste0("Fearon and Laitin (2003) Logit ", FL[[2]]), paste0("Random Forest ", RF[[2]]),  paste0("BART ", BT[[2]])), lty=c(1,2,3), bty="n",
       cex = .75)
dev.off()

# save predicted probabilities from 3 models into a dataset together with actual war onset
data.full$warstds <- ifelse(data.full$warstds == "peace", 0, 1)
out.pred <- as.data.frame(cbind(data.full$warstds, FL.1.pred$war,  RF.1.pred$war,  BT.1.pred$war))
names(out.pred) <- c("onset", "logit", "RandomF", "BART")

#######################################
# produce marginal effects for growth #
#######################################

quant_prob <- c(.01, seq(0.05, 0.95, 0.05), .99)

# marginal effects plot of gdpgrowth
create_test_data <- function(choice){
  if (choice == "high"){
    covariates <- data.full[,-1]
    x <- covariates[which(data.full$warstds == 1),]
  }
  if (choice == "low"){
    covariates <- data.full[,-1]
    x <- covariates[which(data.full$warstds == 0),]
  }

  q <- apply(x[,1:length(covariates)], 2, median)
  test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
  test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
  test <- as.data.frame(test)
  names(test) <- names(covariates)
  return(test)
}

test <- lapply(c("high", "low"), create_test_data)

btmchine_phat <- predict(model.bt, test[[1]], type = "prob")$peace
btmchine.hpc <- 1 - calc_credible_intervals(model.bt$finalModel, test[[1]])
btmchine.plot.data <- as.data.frame(cbind(btmchine_phat, btmchine.hpc[,2], btmchine.hpc[,1], test[[1]]$gdpgrowth))
names(btmchine.plot.data) <- c("p_hat", "ci_lower_bd", "ci_upper_bd", "gdpgrowth")



create_marginal_plot_gdpgrowth <- function(df){
  g <- ggplot(df, aes(x = gdpgrowth, y = p_hat))
  g <- g + geom_line() +  ylim(0, 1)
  g <- g + geom_errorbar(aes(ymax = ci_upper_bd, ymin = ci_lower_bd), width = 0.02)
  g <- g + theme_bw() + theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5))
  g <- g + xlab("GDPgrowth") + ylab("Probability")
  g
}

btmchine.plot <- create_marginal_plot_gdpgrowth(btmchine.plot.data)


data.full$warstds<-factor(
  data.full$warstds,
  levels=c(0,1),
  labels=c("peace", "war"))

model.logit<- train(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
                    +lpopns+coldwar+seceduc+ptime+trade,
                    metric="ROC", method="glm", family="binomial", #uncorrected logistic model
                    trControl=tc, data=data.full)

model.logit


data.full$warstds <- ifelse(data.full$warstds == "peace", 0, 1)

create_test_data_logit <- function(choice){
  if (choice == "high"){
    covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
                               "lpopns", "coldwar", "seceduc", "ptime", "trade")]
    x <- covariates[which(data.full$warstds == 1),]
  }
  if (choice == "low"){
    covariates <- data.full[,c("sxpnew", "sxpsq", "ln_gdpen", "gdpgrowth", "warhist", "lmtnest", "ef", "popdense",
                               "lpopns", "coldwar", "seceduc", "ptime", "trade")]
    x <- covariates[which(data.full$warstds == 0),]
  }

  q <- apply(x[,1:length(covariates)], 2, median)
  test <- apply(matrix(q, nrow = 1), 2, function(x) rep(x, length(quant_prob)))
  test[,which(names(covariates) == "gdpgrowth")] <- quantile(covariates[,which(names(covariates) == "gdpgrowth")], probs = quant_prob)
  test <- as.data.frame(test)
  names(test) <- names(covariates)
  return(test)
}

test <- lapply(c("high", "low"), create_test_data_logit)


summary(model.logit$finalModel)

logit.pred <- predict(model.logit$finalModel, newdata = test[[1]], type = "response", se.fit = T)
ci_logit_upper <- logit.pred$fit + 1.96*logit.pred$se.fit
ci_logit_lower <- logit.pred$fit - 1.96*logit.pred$se.fit


logit.plot.data <- as.data.frame(cbind(ci_logit_upper, ci_logit_lower, logit.pred$fit, test[[1]]$gdpgrowth))
names(logit.plot.data) <- c("ci_upper_bd", "ci_lower_bd", "p_hat", "gdpgrowth")
logit.plot <- create_marginal_plot_gdpgrowth(logit.plot.data)

pp <- grid.arrange(btmchine.plot, logit.plot, ncol = 2)

ggsave(paste0(script.path, "/logit_vs_bart_marginal_gdpgrowth_10_fold.pdf"), pp, height = 8, width = 16)

# save all the data files in the "dbart_mid" dataset
setwd(data.path)
save(model.bt, file = "model.bt.10.fold.RData")
write.csv(out.pred, file = "onset.prediction.10.fold.CSV")


pdf(paste0(script.path, "/partial_growth_10fold.pdf"))
pd_plot(model.bt, "gdpgrowth")
dev.off()



