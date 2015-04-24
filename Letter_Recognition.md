# UCI ML Repo::Letter Recognition: letter classification
bdanalytics  

**  **    
**Date: (Fri) Apr 24, 2015**    

# Introduction:  

Data: 
Source: 
    Training:   https://courses.edx.org/c4x/MITx/15.071x_2/asset/letters_ABPR.csv  
    New:        <newdt_url>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - move model_type parameter to myfit_mdl before indep_vars_vctr (keep all model_* together)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?
- Add custom model to caret for a dummy (baseline) classifier (binomial & multinomial) that generates proba/outcomes which mimics the freq distribution of glb_rsp_var values; Right now glb_dmy_glm_mdl always generates most frequent outcome in training data
- glm_dmy_mdl should use the same method as glm_sel_mdl until custom dummy classifer is implemented

- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Replicate myfit_mdl_classification features in myfit_mdl_regression
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors=FALSE)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
# Gather all package requirements here
#suppressPackageStartupMessages(require())
#packageVersion("snow")

#require(sos); findFn("pinv", maxPages=2, sortby="MaxScore")

# Analysis control global variables
glb_trnng_url <- "https://courses.edx.org/c4x/MITx/15.071x_2/asset/letters_ABPR.csv"
glb_newdt_url <- "<newdt_url>"
glb_is_separate_newent_dataset <- FALSE    # or TRUE
glb_split_entity_newent_datasets <- TRUE   # or FALSE
glb_split_newdata_method <- "sample"          # "condition" or "sample"
glb_split_newdata_condition <- "<col_name> <condition_operator> <value>"    # or NULL
glb_split_newdata_size_ratio <- 0.5               # > 0 & < 1
glb_split_sample.seed <- 2000               # or any integer
glb_max_obs <- NULL # or any integer

glb_is_regression <- FALSE; glb_is_classification <- TRUE; glb_is_binomial <- FALSE

glb_rsp_var_raw <- "letter"

# for classification, the response variable has to be a factor
glb_rsp_var <- "letter.fctr"

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    #relevel(factor(ifelse(raw == "B", "Y", "N")), as.factor(c("Y", "N")), ref="N")
    #as.factor(paste0("B", raw))
    as.factor(raw)    
}
glb_map_rsp_raw_to_var(c("A", "B", "P", "R"))
```

```
## [1] A B P R
## Levels: A B P R
```

```r
glb_map_rsp_var_to_raw <- function(var) {
    levels(var)[as.numeric(var)]
    #as.numeric(var)
}
glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c("A", "B", "P", "R")))
```

```
## [1] "A" "B" "P" "R"
```

```r
if ((glb_rsp_var != glb_rsp_var_raw) & is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

glb_rsp_var_out <- paste0(glb_rsp_var, ".predict.") # model_id is appended later
glb_id_vars <- NULL # or c("<id_var>")

# List transformed vars  
glb_exclude_vars_as_features <- c(NULL) # or c("<var_name>")    
# List feats that shd be excluded due to known causation by prediction variable
if (glb_rsp_var_raw != glb_rsp_var)
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                            glb_rsp_var_raw)
glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                      c(NULL)) # or c("<col_name>")
# List output vars (useful during testing in console)          
# glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
#                         grep(glb_rsp_var_out, names(glb_trnent_df), value=TRUE)) 

glb_impute_na_data <- FALSE            # or TRUE
glb_mice_complete.seed <- 144               # or any integer

# rpart:  .rnorm messes with the models badly
#         caret creates dummy vars for factor feats which messes up the tuning
#             - better to feed as.numeric(<feat>.fctr) to caret 
# Regression
if (glb_is_regression)
    glb_models_method_vctr <- c("lm", "glm", "rpart", "rf") else
# Classification
    if (glb_is_binomial)
        glb_models_method_vctr <- c("glm", "rpart", "rf") else  
        glb_models_method_vctr <- c("rpart", "rf")

glb_models_lst <- list(); glb_models_df <- data.frame()
# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<col_name>")

glb_model_metric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glb_model_metric <- NULL # or "<metric_name>"
glb_model_metric_maximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glb_model_metric_smmry <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glb_model_metric_terms)
#     metric <- sum(confusion_mtrx * glb_model_metric_terms) / nrow(data)
#     names(metric) <- glb_model_metric
#     return(metric)
# }

glb_tune_models_df <- 
   rbind(
    #data.frame(parameter="cp", min=0.00005, max=0.00005, by=0.000005),
                            #seq(from=0.01,  to=0.01, by=0.01)
    #data.frame(parameter="mtry", min=2, max=4, by=1),
    data.frame(parameter="dummy", min=2, max=4, by=1)
        ) 
# or NULL
glb_n_cv_folds <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glb_model_evl_criteria <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit")
if (glb_is_classification) {
    if (glb_is_binomial)
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB", "min.aic.fit") else
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

glb_sel_mdl_id <- NULL # or "<model_id_prefix>.<model_method>"
glb_fin_mdl_id <- glb_sel_mdl_id # or "Final"

# Depict process
glb_analytics_pn <- petrinet(name="glb_analytics_pn",
                        trans_df=data.frame(id=1:6,
    name=c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df=data.frame(
    begin=c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end  =c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](Letter_Recognition_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_script_tm <- proc.time()
glb_script_df <- data.frame(chunk_label="import_data", 
                            chunk_step_major=1, chunk_step_minor=0,
                            elapsed=(proc.time() - glb_script_tm)["elapsed"])
print(tail(glb_script_df, 2))
```

```
##         chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed import_data                1                0   0.002
```

## Step `1`: import data

```r
glb_entity_df <- myimport_data(url=glb_trnng_url, 
    comment=ifelse(!glb_is_separate_newent_dataset, "glb_entity_df", "glb_trnent_df"), 
                                force_header=TRUE)
```

```
## [1] "Reading file ./data/letters_ABPR.csv..."
## [1] "dimensions of data in ./data/letters_ABPR.csv: 3,116 rows x 17 cols"
##   letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 1      B    4    2     5      4     4    8    7     6     6     7      6
## 2      A    1    1     3      2     1    8    2     2     2     8      2
## 3      R    5    9     5      7     6    6   11     7     3     7      3
## 4      B    5    9     7      7    10    9    8     4     4     6      8
## 5      P    3    6     4      4     2    4   14     8     1    11      6
## 6      R    8   10     8      6     6    7    7     3     5     8      4
##   xy2bar xedge xedgeycor yedge yedgexcor
## 1      6     2         8     7        10
## 2      8     1         6     2         7
## 3      9     2         7     5        11
## 4      6     6        11     8         7
## 5      3     0        10     4         8
## 6      8     6         6     7         7
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 108       P    5    4     5      6     3    4   12     9     3    10
## 475       P    4    9     4      6     4    5   10     8     2     9
## 1013      R    6    8     9      7     9    9    6     4     4     8
## 1587      P    2    1     3      1     1    5    9     4     4     9
## 2267      P    5    9     7      7     4    9    7     3     6    13
## 3082      B    4    9     4      7     6    6    8     8     5     7
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 108       6      4     1        10     4         8
## 475       6      5     1        10     3         8
## 1013      5      7     7         9     7         6
## 1587      7      4     1         9     3         7
## 2267      4      5     5        10     5        10
## 3082      5      7     2         8     7         9
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 3111      B    4    8     6      6     5    7    8     6     6    10
## 3112      A    2    3     3      1     1    6    2     2     1     5
## 3113      A    3    9     5      6     2    6    5     3     1     6
## 3114      R    2    3     3      2     2    7    7     5     5     7
## 3115      P    2    1     3      2     1    4   10     3     5    10
## 3116      A    4    9     6      6     2    9    5     3     1     8
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 3111      6      6     3         8     7         8
## 3112      2      8     1         6     1         7
## 3113      1      8     2         7     2         7
## 3114      5      6     2         7     4         8
## 3115      8      5     0         9     3         7
## 3116      1      8     2         7     2         8
## 'data.frame':	3116 obs. of  17 variables:
##  $ letter   : chr  "B" "A" "R" "B" ...
##  $ xbox     : int  4 1 5 5 3 8 2 3 8 6 ...
##  $ ybox     : int  2 1 9 9 6 10 6 7 14 10 ...
##  $ width    : int  5 3 5 7 4 8 4 5 7 8 ...
##  $ height   : int  4 2 7 7 4 6 4 5 8 8 ...
##  $ onpix    : int  4 1 6 10 2 6 3 3 4 7 ...
##  $ xbar     : int  8 8 6 9 4 7 6 12 5 8 ...
##  $ ybar     : int  7 2 11 8 14 7 7 2 10 5 ...
##  $ x2bar    : int  6 2 7 4 8 3 5 3 6 7 ...
##  $ y2bar    : int  6 2 3 4 1 5 5 2 3 5 ...
##  $ xybar    : int  7 8 7 6 11 8 6 10 12 7 ...
##  $ x2ybar   : int  6 2 3 8 6 4 5 2 5 6 ...
##  $ xy2bar   : int  6 8 9 6 3 8 7 9 4 6 ...
##  $ xedge    : int  2 1 2 6 0 6 3 2 4 3 ...
##  $ xedgeycor: int  8 6 7 11 10 6 7 6 10 9 ...
##  $ yedge    : int  7 2 5 8 4 7 5 3 4 8 ...
##  $ yedgexcor: int  10 7 11 7 8 7 8 8 8 9 ...
##  - attr(*, "comment")= chr "glb_entity_df"
## NULL
```

```r
if (!glb_is_separate_newent_dataset) {
    glb_trnent_df <- glb_entity_df; comment(glb_trnent_df) <- "glb_trnent_df"
} # else glb_entity_df is maintained as is for chunk:inspectORexplore.data
    
if (glb_is_separate_newent_dataset) {
    glb_newent_df <- myimport_data(
        url=glb_newdt_url, 
        comment="glb_newent_df", force_header=TRUE)
    
    # To make plots / stats / checks easier in chunk:inspectORexplore.data
    glb_entity_df <- rbind(glb_trnent_df, glb_newent_df); comment(glb_entity_df) <- "glb_entity_df"
} else {
    if (!glb_split_entity_newent_datasets) {
        stop("Not implemented yet") 
        glb_newent_df <- glb_trnent_df[sample(1:nrow(glb_trnent_df),
                                          max(2, nrow(glb_trnent_df) / 1000)),]                    
    } else      if (glb_split_newdata_method == "condition") {
            glb_newent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=glb_split_newdata_condition)))
            glb_trnent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=paste0("!(", 
                                                      glb_split_newdata_condition,
                                                      ")"))))
        } else if (glb_split_newdata_method == "sample") {
                require(caTools)
                
                set.seed(glb_split_sample.seed)
                split <- sample.split(glb_trnent_df[, glb_rsp_var_raw], 
                                      SplitRatio=(1-glb_split_newdata_size_ratio))
                glb_newent_df <- glb_trnent_df[!split, ] 
                glb_trnent_df <- glb_trnent_df[split ,]
        } else stop("glb_split_newdata_method should be %in% c('condition', 'sample')")   

    comment(glb_newent_df) <- "glb_newent_df"
    myprint_df(glb_newent_df)
    str(glb_newent_df)

    if (glb_split_entity_newent_datasets) {
        myprint_df(glb_trnent_df)
        str(glb_trnent_df)        
    }
}         
```

```
## Loading required package: caTools
```

```
##    letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 4       B    5    9     7      7    10    9    8     4     4     6      8
## 7       R    2    6     4      4     3    6    7     5     5     6      5
## 8       A    3    7     5      5     3   12    2     3     2    10      2
## 9       P    8   14     7      8     4    5   10     6     3    12      5
## 11      A    3    8     5      6     3    9    2     2     3     8      2
## 12      R    6    9     5      4     3   10    6     5     5    10      2
##    xy2bar xedge xedgeycor yedge yedgexcor
## 4       6     6        11     8         7
## 7       7     3         7     5         8
## 8       9     2         6     3         8
## 9       4     4        10     4         8
## 11      8     2         6     3         7
## 12      8     6         6     4         9
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 62        A    2    6     4      4     2   12    2     4     3    12
## 573       R    4    7     4      4     2    5   11     8     3     7
## 1285      A    2    3     3      2     1   10    2     3     1    10
## 2256      P    7   10    10      8     6    8   10     7     5     9
## 2739      P    2    4     4      3     2    6   10     3     4    12
## 3018      R    3    4     4      6     3    5   11     8     4     7
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 62        2     10     2         6     3         9
## 573       4      8     3         7     6        11
## 1285      2      9     2         6     2         8
## 2256      4      3     3        10     5         9
## 2739      5      3     1        10     2         8
## 3018      2      9     3         7     6        11
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 3106      A    3    7     4      5     2    7    4     2     0     6
## 3107      P    4    6     5      4     4    8    5     6     5     7
## 3109      A    6   10     7      6     3   12    0     4     1    11
## 3110      P    2    5     3      7     5    8    6     5     1     7
## 3114      R    2    3     3      2     2    7    7     5     5     7
## 3116      A    4    9     6      6     2    9    5     3     1     8
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 3106      2      8     2         7     1         7
## 3107      6      7     2         9     7         9
## 3109      4     12     4         4     3        11
## 3110      6      7     6         8     5         9
## 3114      5      6     2         7     4         8
## 3116      1      8     2         7     2         8
## 'data.frame':	1558 obs. of  17 variables:
##  $ letter   : chr  "B" "R" "A" "P" ...
##  $ xbox     : int  5 2 3 8 3 6 2 2 2 4 ...
##  $ ybox     : int  9 6 7 14 8 9 1 4 8 2 ...
##  $ width    : int  7 4 5 7 5 5 4 4 4 5 ...
##  $ height   : int  7 4 5 8 6 4 2 3 6 4 ...
##  $ onpix    : int  10 3 3 4 3 3 1 2 2 4 ...
##  $ xbar     : int  9 6 12 5 9 10 8 10 12 7 ...
##  $ ybar     : int  8 7 2 10 2 6 1 2 2 7 ...
##  $ x2bar    : int  4 5 3 6 2 5 2 2 4 5 ...
##  $ y2bar    : int  4 5 2 3 3 5 2 2 3 6 ...
##  $ xybar    : int  6 6 10 12 8 10 7 9 11 7 ...
##  $ x2ybar   : int  8 5 2 5 2 2 2 2 2 6 ...
##  $ xy2bar   : int  6 7 9 4 8 8 8 9 10 6 ...
##  $ xedge    : int  6 3 2 4 2 6 2 2 3 2 ...
##  $ xedgeycor: int  11 7 6 10 6 6 5 7 6 8 ...
##  $ yedge    : int  8 5 3 4 3 4 2 3 3 7 ...
##  $ yedgexcor: int  7 8 8 8 7 9 7 9 9 10 ...
##  - attr(*, "comment")= chr "glb_newent_df"
##    letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 1       B    4    2     5      4     4    8    7     6     6     7      6
## 2       A    1    1     3      2     1    8    2     2     2     8      2
## 3       R    5    9     5      7     6    6   11     7     3     7      3
## 5       P    3    6     4      4     2    4   14     8     1    11      6
## 6       R    8   10     8      6     6    7    7     3     5     8      4
## 10      P    6   10     8      8     7    8    5     7     5     7      6
##    xy2bar xedge xedgeycor yedge yedgexcor
## 1       6     2         8     7        10
## 2       8     1         6     2         7
## 3       9     2         7     5        11
## 5       3     0        10     4         8
## 6       8     6         6     7         7
## 10      6     3         9     8         9
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 10        P    6   10     8      8     7    8    5     7     5     7
## 782       P    6   11     9      8     7    9    8     2     5    13
## 793       A    5   11     5      6     4    9    4     5     3    10
## 1674      B    8   15     6      8     5    8    6     5     6    10
## 1710      B    4    7     6      6     7    8    6     5     4     7
## 2401      B    3    5     4      3     3    7    7     5     5     6
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 10        6      6     3         9     8         9
## 782       5      5     3         9     4         9
## 793       6     12     7         4     6        10
## 1674      5      9     6         7     8        10
## 1710      6      8     6         9     7         7
## 2401      6      6     2         8     6        10
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 3105      A    2    2     4      4     2    8    2     1     2     7
## 3108      P    4    9     6      6     4    7    9     3     4    12
## 3111      B    4    8     6      6     5    7    8     6     6    10
## 3112      A    2    3     3      1     1    6    2     2     1     5
## 3113      A    3    9     5      6     2    6    5     3     1     6
## 3115      P    2    1     3      2     1    4   10     3     5    10
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 3105      2      8     2         7     3         7
## 3108      5      4     2         9     3         8
## 3111      6      6     3         8     7         8
## 3112      2      8     1         6     1         7
## 3113      1      8     2         7     2         7
## 3115      8      5     0         9     3         7
## 'data.frame':	1558 obs. of  17 variables:
##  $ letter   : chr  "B" "A" "R" "P" ...
##  $ xbox     : int  4 1 5 3 8 6 3 5 3 7 ...
##  $ ybox     : int  2 1 9 6 10 10 3 9 7 11 ...
##  $ width    : int  5 3 5 4 8 8 3 7 5 10 ...
##  $ height   : int  4 2 7 4 6 8 4 7 5 8 ...
##  $ onpix    : int  4 1 6 2 6 7 3 7 3 6 ...
##  $ xbar     : int  8 8 6 4 7 8 7 8 10 6 ...
##  $ ybar     : int  7 2 11 14 7 5 7 8 4 11 ...
##  $ x2bar    : int  6 2 7 8 3 7 5 3 1 3 ...
##  $ y2bar    : int  6 2 3 1 5 5 5 6 2 6 ...
##  $ xybar    : int  7 8 7 11 8 7 7 10 8 13 ...
##  $ x2ybar   : int  6 2 3 6 4 6 6 5 3 6 ...
##  $ xy2bar   : int  6 8 9 3 8 6 6 6 9 3 ...
##  $ xedge    : int  2 1 2 0 6 3 5 3 2 0 ...
##  $ xedgeycor: int  8 6 7 10 6 9 8 7 4 10 ...
##  $ yedge    : int  7 2 5 4 7 8 5 6 2 3 ...
##  $ yedgexcor: int  10 7 11 8 7 9 10 8 7 8 ...
##  - attr(*, "comment")= chr "glb_trnent_df"
```

```r
if (!is.null(glb_max_obs)) {
    if (nrow(glb_trnent_df) > glb_max_obs) {
        warning("glb_trnent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))
        org_entity_df <- glb_trnent_df
        glb_trnent_df <- org_entity_df[split <- 
            sample.split(org_entity_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_entity_df <- NULL
    }
    if (nrow(glb_newent_df) > glb_max_obs) {
        warning("glb_newent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))        
        org_newent_df <- glb_newent_df
        glb_newent_df <- org_newent_df[split <- 
            sample.split(org_newent_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_newent_df <- NULL
    }    
}

glb_script_df <- rbind(glb_script_df,
                   data.frame(chunk_label="cleanse_data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##           chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed   import_data                1                0   0.002
## elapsed1 cleanse_data                2                0   0.548
```

## Step `2`: cleanse data

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="inspectORexplore.data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major), 
                              chunk_step_minor=1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed1          cleanse_data                2                0   0.548
## elapsed2 inspectORexplore.data                2                1   0.580
```

### Step `2`.`1`: inspect/explore data

```r
#print(str(glb_trnent_df))
#View(glb_trnent_df)

# List info gathered for various columns
# <col_name>:   <description>; <notes>
# letter = the letter that the image corresponds to (A, B, P or R)
# xbox = the horizontal position of where the smallest box covering the letter shape begins.
# ybox = the vertical position of where the smallest box covering the letter shape begins.
# width = the width of this smallest box.
# height = the height of this smallest box.
# onpix = the total number of "on" pixels in the character image
# xbar = the mean horizontal position of all of the "on" pixels
# ybar = the mean vertical position of all of the "on" pixels
# x2bar = the mean squared horizontal position of all of the "on" pixels in the image
# y2bar = the mean squared vertical position of all of the "on" pixels in the image
# xybar = the mean of the product of the horizontal and vertical position of all of the "on" pixels in the image
# x2ybar = the mean of the product of the squared horizontal position and the vertical position of all of the "on" pixels
# xy2bar = the mean of the product of the horizontal position and the squared vertical position of all of the "on" pixels
# xedge = the mean number of edges (the number of times an "off" pixel is followed by an "on" pixel, or the image boundary is hit) as the image is scanned from left to right, along the whole vertical length of the image
# xedgeycor = the mean of the product of the number of horizontal edges at each vertical position and the vertical position
# yedge = the mean number of edges as the images is scanned from top to bottom, along the whole horizontal length of the image
# yedgexcor = the mean of the product of the number of vertical edges at each horizontal position and the horizontal position

# Create new features that help diagnostics
#   Create factors of string variables
str_vars <- sapply(1:length(names(glb_trnent_df)), 
    function(col) ifelse(class(glb_trnent_df[, names(glb_trnent_df)[col]]) == "character",
                         names(glb_trnent_df)[col], ""))
if (length(str_vars <- setdiff(str_vars[str_vars != ""], 
                               glb_exclude_vars_as_features)) > 0) {
    warning("Creating factors of string variables:", paste0(str_vars, collapse=", "))
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, str_vars)
    for (var in str_vars) {
        glb_entity_df[, paste0(var, ".fctr")] <- factor(glb_entity_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_trnent_df[, paste0(var, ".fctr")] <- factor(glb_trnent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_newent_df[, paste0(var, ".fctr")] <- factor(glb_newent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
    }
}

#   Convert factors to dummy variables
#   Build splines   require(splines); bsBasis <- bs(training$age, df=3)

add_new_diag_feats <- function(obs_df, ref_df=glb_entity_df) {
    require(plyr)
    
    obs_df <- mutate(obs_df,
#         <col_name>.NA=is.na(<col_name>),

#         <col_name>.fctr=factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))), 
#         <col_name>.fctr=relevel(factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))),
#                                   "<ref_val>"), 
#         <col2_name>.fctr=relevel(factor(ifelse(<col1_name> == <val>, "<oth_val>", "<ref_val>")), 
#                               as.factor(c("R", "<ref_val>")),
#                               ref="<ref_val>"),

          # This doesn't work - use sapply instead
#         <col_name>.fctr_num=grep(<col_name>, levels(<col_name>.fctr)), 
#         
#         Date.my=as.Date(strptime(Date, "%m/%d/%y %H:%M")),
#         Year=year(Date.my),
#         Month=months(Date.my),
#         Weekday=weekdays(Date.my)

#         <col_name>.log=log(<col.name>),        
#         <col_name>=<table>[as.character(<col2_name>)],
#         <col_name>=as.numeric(<col2_name>),

        .rnorm=rnorm(n=nrow(obs_df))
                        )

    # If levels of a factor are different across obs_df & glb_newent_df; predict.glm fails  
    # Transformations not handled by mutate
#     obs_df$<col_name>.fctr.num <- sapply(1:nrow(obs_df), 
#         function(row_ix) grep(obs_df[row_ix, "<col_name>"],
#                               levels(obs_df[row_ix, "<col_name>.fctr"])))
    
    print(summary(obs_df))
    print(sapply(names(obs_df), function(col) sum(is.na(obs_df[, col]))))
    return(obs_df)
}

glb_entity_df <- add_new_diag_feats(glb_entity_df)
```

```
## Loading required package: plyr
```

```
##     letter               xbox             ybox            width       
##  Length:3116        Min.   : 0.000   Min.   : 0.000   Min.   : 1.000  
##  Class :character   1st Qu.: 3.000   1st Qu.: 5.000   1st Qu.: 4.000  
##  Mode  :character   Median : 4.000   Median : 7.000   Median : 5.000  
##                     Mean   : 3.915   Mean   : 7.051   Mean   : 5.186  
##                     3rd Qu.: 5.000   3rd Qu.: 9.000   3rd Qu.: 6.000  
##                     Max.   :13.000   Max.   :15.000   Max.   :11.000  
##      height           onpix             xbar             ybar       
##  Min.   : 0.000   Min.   : 0.000   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 4.000   1st Qu.: 2.000   1st Qu.: 6.000   1st Qu.: 6.000  
##  Median : 6.000   Median : 4.000   Median : 7.000   Median : 7.000  
##  Mean   : 5.276   Mean   : 3.869   Mean   : 7.469   Mean   : 7.197  
##  3rd Qu.: 7.000   3rd Qu.: 5.000   3rd Qu.: 8.000   3rd Qu.: 9.000  
##  Max.   :12.000   Max.   :12.000   Max.   :14.000   Max.   :15.000  
##      x2bar            y2bar           xybar            x2ybar     
##  Min.   : 0.000   Min.   :0.000   Min.   : 3.000   Min.   : 0.00  
##  1st Qu.: 3.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.: 3.00  
##  Median : 4.000   Median :4.000   Median : 8.000   Median : 5.00  
##  Mean   : 4.706   Mean   :3.903   Mean   : 8.491   Mean   : 4.52  
##  3rd Qu.: 6.000   3rd Qu.:5.000   3rd Qu.:10.000   3rd Qu.: 6.00  
##  Max.   :11.000   Max.   :8.000   Max.   :14.000   Max.   :10.00  
##      xy2bar           xedge          xedgeycor          yedge     
##  Min.   : 0.000   Min.   : 0.000   Min.   : 1.000   Min.   : 0.0  
##  1st Qu.: 6.000   1st Qu.: 2.000   1st Qu.: 7.000   1st Qu.: 3.0  
##  Median : 7.000   Median : 2.000   Median : 8.000   Median : 4.0  
##  Mean   : 6.711   Mean   : 2.913   Mean   : 7.763   Mean   : 4.6  
##  3rd Qu.: 8.000   3rd Qu.: 4.000   3rd Qu.: 9.000   3rd Qu.: 6.0  
##  Max.   :14.000   Max.   :10.000   Max.   :13.000   Max.   :12.0  
##    yedgexcor          .rnorm         
##  Min.   : 1.000   Min.   :-3.587688  
##  1st Qu.: 7.000   1st Qu.:-0.654822  
##  Median : 8.000   Median :-0.026394  
##  Mean   : 8.418   Mean   :-0.008335  
##  3rd Qu.:10.000   3rd Qu.: 0.663858  
##  Max.   :13.000   Max.   : 3.864162  
##    letter      xbox      ybox     width    height     onpix      xbar 
##         0         0         0         0         0         0         0 
##      ybar     x2bar     y2bar     xybar    x2ybar    xy2bar     xedge 
##         0         0         0         0         0         0         0 
## xedgeycor     yedge yedgexcor    .rnorm 
##         0         0         0         0
```

```r
glb_trnent_df <- add_new_diag_feats(glb_trnent_df)
```

```
##     letter               xbox             ybox            width       
##  Length:1558        Min.   : 0.000   Min.   : 0.000   Min.   : 1.000  
##  Class :character   1st Qu.: 3.000   1st Qu.: 5.000   1st Qu.: 4.000  
##  Mode  :character   Median : 4.000   Median : 7.000   Median : 5.000  
##                     Mean   : 3.917   Mean   : 7.074   Mean   : 5.161  
##                     3rd Qu.: 5.000   3rd Qu.:10.000   3rd Qu.: 6.000  
##                     Max.   :13.000   Max.   :15.000   Max.   :10.000  
##      height           onpix             xbar             ybar       
##  Min.   : 0.000   Min.   : 0.000   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 4.000   1st Qu.: 2.000   1st Qu.: 6.000   1st Qu.: 6.000  
##  Median : 6.000   Median : 4.000   Median : 7.000   Median : 7.000  
##  Mean   : 5.273   Mean   : 3.828   Mean   : 7.432   Mean   : 7.226  
##  3rd Qu.: 7.000   3rd Qu.: 5.000   3rd Qu.: 8.000   3rd Qu.: 9.000  
##  Max.   :12.000   Max.   :12.000   Max.   :14.000   Max.   :15.000  
##      x2bar            y2bar           xybar            x2ybar     
##  Min.   : 0.000   Min.   :0.000   Min.   : 3.000   Min.   : 0.00  
##  1st Qu.: 3.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.: 3.00  
##  Median : 4.000   Median :4.000   Median : 8.000   Median : 5.00  
##  Mean   : 4.733   Mean   :3.863   Mean   : 8.484   Mean   : 4.49  
##  3rd Qu.: 6.000   3rd Qu.:5.000   3rd Qu.:10.000   3rd Qu.: 6.00  
##  Max.   :11.000   Max.   :8.000   Max.   :14.000   Max.   :10.00  
##      xy2bar           xedge         xedgeycor          yedge       
##  Min.   : 0.000   Min.   :0.000   Min.   : 1.000   Min.   : 0.000  
##  1st Qu.: 6.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.: 3.000  
##  Median : 7.000   Median :2.000   Median : 8.000   Median : 4.000  
##  Mean   : 6.722   Mean   :2.885   Mean   : 7.746   Mean   : 4.585  
##  3rd Qu.: 8.000   3rd Qu.:4.000   3rd Qu.: 9.000   3rd Qu.: 6.000  
##  Max.   :12.000   Max.   :9.000   Max.   :12.000   Max.   :11.000  
##    yedgexcor          .rnorm        
##  Min.   : 1.000   Min.   :-3.04339  
##  1st Qu.: 7.000   1st Qu.:-0.73679  
##  Median : 8.000   Median :-0.02599  
##  Mean   : 8.397   Mean   :-0.02950  
##  3rd Qu.:10.000   3rd Qu.: 0.66800  
##  Max.   :13.000   Max.   : 3.18902  
##    letter      xbox      ybox     width    height     onpix      xbar 
##         0         0         0         0         0         0         0 
##      ybar     x2bar     y2bar     xybar    x2ybar    xy2bar     xedge 
##         0         0         0         0         0         0         0 
## xedgeycor     yedge yedgexcor    .rnorm 
##         0         0         0         0
```

```r
glb_newent_df <- add_new_diag_feats(glb_newent_df)
```

```
##     letter               xbox             ybox            width       
##  Length:1558        Min.   : 1.000   Min.   : 0.000   Min.   : 1.000  
##  Class :character   1st Qu.: 3.000   1st Qu.: 5.000   1st Qu.: 4.000  
##  Mode  :character   Median : 4.000   Median : 7.000   Median : 5.000  
##                     Mean   : 3.913   Mean   : 7.029   Mean   : 5.212  
##                     3rd Qu.: 5.000   3rd Qu.: 9.000   3rd Qu.: 6.000  
##                     Max.   :12.000   Max.   :15.000   Max.   :11.000  
##      height           onpix             xbar             ybar       
##  Min.   : 0.000   Min.   : 0.000   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 4.000   1st Qu.: 2.000   1st Qu.: 6.000   1st Qu.: 6.000  
##  Median : 6.000   Median : 4.000   Median : 7.000   Median : 7.000  
##  Mean   : 5.279   Mean   : 3.909   Mean   : 7.505   Mean   : 7.168  
##  3rd Qu.: 7.000   3rd Qu.: 5.000   3rd Qu.: 9.000   3rd Qu.: 9.000  
##  Max.   :12.000   Max.   :12.000   Max.   :14.000   Max.   :15.000  
##      x2bar           y2bar           xybar            x2ybar     
##  Min.   : 1.00   Min.   :0.000   Min.   : 3.000   Min.   :0.000  
##  1st Qu.: 3.00   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.:3.000  
##  Median : 4.00   Median :4.000   Median : 8.000   Median :5.000  
##  Mean   : 4.68   Mean   :3.944   Mean   : 8.499   Mean   :4.549  
##  3rd Qu.: 6.00   3rd Qu.:5.000   3rd Qu.:10.000   3rd Qu.:6.000  
##  Max.   :11.00   Max.   :8.000   Max.   :14.000   Max.   :9.000  
##      xy2bar         xedge          xedgeycor         yedge       
##  Min.   : 1.0   Min.   : 0.000   Min.   : 2.00   Min.   : 1.000  
##  1st Qu.: 6.0   1st Qu.: 2.000   1st Qu.: 7.00   1st Qu.: 3.000  
##  Median : 7.0   Median : 2.000   Median : 8.00   Median : 4.000  
##  Mean   : 6.7   Mean   : 2.942   Mean   : 7.78   Mean   : 4.616  
##  3rd Qu.: 8.0   3rd Qu.: 4.000   3rd Qu.: 9.00   3rd Qu.: 6.000  
##  Max.   :14.0   Max.   :10.000   Max.   :13.00   Max.   :12.000  
##    yedgexcor          .rnorm        
##  Min.   : 2.000   Min.   :-3.25766  
##  1st Qu.: 7.250   1st Qu.:-0.72221  
##  Median : 8.000   Median :-0.07262  
##  Mean   : 8.438   Mean   :-0.03707  
##  3rd Qu.:10.000   3rd Qu.: 0.66899  
##  Max.   :12.000   Max.   : 2.92062  
##    letter      xbox      ybox     width    height     onpix      xbar 
##         0         0         0         0         0         0         0 
##      ybar     x2bar     y2bar     xybar    x2ybar    xy2bar     xedge 
##         0         0         0         0         0         0         0 
## xedgeycor     yedge yedgexcor    .rnorm 
##         0         0         0         0
```

```r
# Histogram of predictor in glb_trnent_df & glb_newent_df
plot_df <- rbind(cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="Training")),
                 cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="New")))
print(myplot_histogram(plot_df, glb_rsp_var_raw) + facet_wrap(~ .data))
```

```
## Warning in mean.default(sort(x, partial = half + 0L:1L)[half + 0L:1L]):
## argument is not numeric or logical: returning NA
```

```
## Warning: Removed 1 rows containing missing values (geom_segment).
```

```
## Warning: Removed 1 rows containing missing values (geom_segment).
```

![](Letter_Recognition_files/figure-html/inspectORexplore_data-1.png) 

```r
if (glb_is_classification) {
    xtab_df <- mycreate_xtab(plot_df, c(".data", glb_rsp_var_raw))
    rownames(xtab_df) <- xtab_df$.data
    xtab_df <- subset(xtab_df, select=-.data)
    print(xtab_df / rowSums(xtab_df))    
}    
```

```
## Loading required package: reshape2
```

```
##           letter.A letter.B  letter.P  letter.R
## New      0.2528883 0.245828 0.2580231 0.2432606
## Training 0.2528883 0.245828 0.2580231 0.2432606
```

```r
# Check for duplicates in glb_id_vars
if (length(glb_id_vars) > 0) {
    id_vars_dups_df <- subset(id_vars_df <- 
            mycreate_tbl_df(glb_entity_df[, glb_id_vars, FALSE], glb_id_vars),
                                .freq > 1)
    if (nrow(id_vars_dups_df) > 0) {
        warning("Duplicates found in glb_id_vars data:", nrow(id_vars_dups_df))
        myprint_df(id_vars_dups_df)
    } else {
        # glb_id_vars are unique across obs in both glb_<>_df
        glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, glb_id_vars)
    }
}

#pairs(subset(glb_trnent_df, select=-c(col_symbol)))
# Check for glb_newent_df & glb_trnent_df features range mismatches

# Other diagnostics:
# print(subset(glb_trnent_df, <col1_name> == max(glb_trnent_df$<col1_name>, na.rm=TRUE) & 
#                         <col2_name> <= mean(glb_trnent_df$<col1_name>, na.rm=TRUE)))

# print(glb_trnent_df[which.max(glb_trnent_df$<col_name>),])

# print(<col_name>_freq_glb_trnent_df <- mycreate_tbl_df(glb_trnent_df, "<col_name>"))
# print(which.min(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>)[, 2]))
# print(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>))
# print(table(is.na(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(table(sign(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(mycreate_xtab(glb_trnent_df, <col1_name>))
# print(mycreate_xtab(glb_trnent_df, c(<col1_name>, <col2_name>)))
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mycreate_xtab(glb_trnent_df, c("<col1_name>", "<col2_name>")))
# <col1_name>_<col2_name>_xtab_glb_trnent_df[is.na(<col1_name>_<col2_name>_xtab_glb_trnent_df)] <- 0
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mutate(<col1_name>_<col2_name>_xtab_glb_trnent_df, 
#             <col3_name>=(<col1_name> * 1.0) / (<col1_name> + <col2_name>))) 

# print(<col2_name>_min_entity_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>, min, na.rm=TRUE)))
# print(<col1_name>_na_by_<col2_name>_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>.NA, glb_trnent_df$<col2_name>, mean, na.rm=TRUE)))

# Other plots:
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>"))
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>", xcol_name="<col2_name>"))
# print(myplot_line(subset(glb_trnent_df, Symbol %in% c("KO", "PG")), 
#                   "Date.my", "StockPrice", facet_row_colnames="Symbol") + 
#     geom_vline(xintercept=as.numeric(as.Date("2003-03-01"))) +
#     geom_vline(xintercept=as.numeric(as.Date("1983-01-01")))        
#         )
# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", colorcol_name="<Pred.fctr>") + 
#         geom_point(data=subset(glb_entity_df, <condition>), 
#                     mapping=aes(x=<x_var>, y=<y_var>), color="red", shape=4, size=5))

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="manage_missing_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed2 inspectORexplore.data                2                1   0.580
## elapsed3   manage_missing_data                2                2   1.543
```

### Step `2`.`2`: manage missing data

```r
# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
# glb_trnent_df <- na.omit(glb_trnent_df)
# glb_newent_df <- na.omit(glb_newent_df)
# df[is.na(df)] <- 0

# Not refactored into mydsutils.R since glb_*_df might be reassigned
glb_impute_missing_data <- function(entity_df, newent_df) {
    if (!glb_is_separate_newent_dataset) {
        # Combine entity & newent
        union_df <- rbind(mutate(entity_df, .src = "entity"),
                          mutate(newent_df, .src = "newent"))
        union_imputed_df <- union_df[, setdiff(setdiff(names(entity_df), 
                                                       glb_rsp_var), 
                                               glb_exclude_vars_as_features)]
        print(summary(union_imputed_df))
    
        require(mice)
        set.seed(glb_mice_complete.seed)
        union_imputed_df <- complete(mice(union_imputed_df))
        print(summary(union_imputed_df))
    
        union_df[, names(union_imputed_df)] <- union_imputed_df[, names(union_imputed_df)]
        print(summary(union_df))
#         union_df$.rownames <- rownames(union_df)
#         union_df <- orderBy(~.rownames, union_df)
#         
#         imp_entity_df <- myimport_data(
#             url="<imputed_trnng_url>", 
#             comment="imp_entity_df", force_header=TRUE, print_diagn=TRUE)
#         print(all.equal(subset(union_df, select=-c(.src, .rownames, .rnorm)), 
#                         imp_entity_df))
        
        # Partition again
        glb_trnent_df <<- subset(union_df, .src == "entity", select=-c(.src, .rownames))
        comment(glb_trnent_df) <- "entity_df"
        glb_newent_df <<- subset(union_df, .src == "newent", select=-c(.src, .rownames))
        comment(glb_newent_df) <- "newent_df"
        
        # Generate summaries
        print(summary(entity_df))
        print(sapply(names(entity_df), function(col) sum(is.na(entity_df[, col]))))
        print(summary(newent_df))
        print(sapply(names(newent_df), function(col) sum(is.na(newent_df[, col]))))
    
    } else stop("Not implemented yet")
}

if (glb_impute_na_data) {
    if ((sum(sapply(names(glb_trnent_df), 
                    function(col) sum(is.na(glb_trnent_df[, col])))) > 0) | 
        (sum(sapply(names(glb_newent_df), 
                    function(col) sum(is.na(glb_newent_df[, col])))) > 0))
        glb_impute_missing_data(glb_trnent_df, glb_newent_df)
}    

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="encode_retype_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed3 manage_missing_data                2                2   1.543
## elapsed4  encode_retype_data                2                3   1.986
```

### Step `2`.`3`: encode/retype data

```r
# map_<col_name>_df <- myimport_data(
#     url="<map_url>", 
#     comment="map_<col_name>_df", print_diagn=TRUE)
# map_<col_name>_df <- read.csv(paste0(getwd(), "/data/<file_name>.csv"), strip.white=TRUE)

# glb_trnent_df <- mymap_codes(glb_trnent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
# glb_newent_df <- mymap_codes(glb_newent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
    					
# glb_trnent_df$<col_name>.fctr <- factor(glb_trnent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))
# glb_newent_df$<col_name>.fctr <- factor(glb_newent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))

if (!is.null(glb_map_rsp_raw_to_var)) {
    glb_entity_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_entity_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_entity_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_trnent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_trnent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_trnent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_newent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_newent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_newent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)    
}
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##   letter letter.fctr  .n
## 1      P           P 803
## 2      A           A 789
## 3      B           B 766
## 4      R           R 758
```

![](Letter_Recognition_files/figure-html/encode_retype_data_1-1.png) 

```
##   letter letter.fctr  .n
## 1      P           P 402
## 2      A           A 394
## 3      B           B 383
## 4      R           R 379
```

![](Letter_Recognition_files/figure-html/encode_retype_data_1-2.png) 

```
##   letter letter.fctr  .n
## 1      P           P 401
## 2      A           A 395
## 3      B           B 383
## 4      R           R 379
```

![](Letter_Recognition_files/figure-html/encode_retype_data_1-3.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="extract_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                 chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed4 encode_retype_data                2                3   1.986
## elapsed5   extract_features                3                0   5.054
```

## Step `3`: extract features

```r
# Create new features that help prediction
# <col_name>.lag.2 <- lag(zoo(glb_trnent_df$<col_name>), -2, na.pad=TRUE)
# glb_trnent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# <col_name>.lag.2 <- lag(zoo(glb_newent_df$<col_name>), -2, na.pad=TRUE)
# glb_newent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# 
# glb_newent_df[1, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df) - 1, 
#                                                    "<col_name>"]
# glb_newent_df[2, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df), 
#                                                    "<col_name>"]
                                                   
# glb_trnent_df <- mutate(glb_trnent_df,
#     <new_col_name>=
#                     )

# glb_newent_df <- mutate(glb_newent_df,
#     <new_col_name>=
#                     )

# print(summary(glb_trnent_df))
# print(summary(glb_newent_df))

# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))

# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all","data.new")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Letter_Recognition_files/figure-html/extract_features-1.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="select_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##               chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed5 extract_features                3                0   5.054
## elapsed6  select_features                4                0   6.513
```

## Step `4`: select features

```r
print(glb_feats_df <- myselect_features(entity_df=glb_trnent_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                  id       cor.y exclude.as.feat  cor.y.abs
## ybar           ybar  0.67198759               0 0.67198759
## xbar           xbar -0.41476375               0 0.41476375
## x2bar         x2bar  0.41019605               0 0.41019605
## x2ybar       x2ybar  0.38607514               0 0.38607514
## y2bar         y2bar  0.33859131               0 0.33859131
## yedgexcor yedgexcor  0.31367199               0 0.31367199
## xy2bar       xy2bar -0.27957030               0 0.27957030
## xedgeycor xedgeycor  0.27453618               0 0.27453618
## yedge         yedge  0.24976884               0 0.24976884
## xbox           xbox  0.16830409               0 0.16830409
## onpix         onpix  0.16721154               0 0.16721154
## xybar         xybar  0.12073750               0 0.12073750
## xedge         xedge  0.11786463               0 0.11786463
## width         width  0.04909820               0 0.04909820
## height       height  0.04565534               0 0.04565534
## ybox           ybox  0.03690669               0 0.03690669
## .rnorm       .rnorm -0.03047401               0 0.03047401
```

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="remove_correlated_features", 
        chunk_step_major=max(glb_script_df$chunk_step_major),
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))        
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed6            select_features                4                0
## elapsed7 remove_correlated_features                4                1
##          elapsed
## elapsed6   6.513
## elapsed7   6.727
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, 
          myfind_cor_features(feats_df=glb_feats_df, entity_df=glb_trnent_df, 
                                rsp_var=glb_rsp_var)))
```

```
##                  ybar         xbar       x2bar      x2ybar        y2bar
## ybar       1.00000000 -0.688602938  0.57264414  0.60459376  0.237523198
## xbar      -0.68860294  1.000000000 -0.54533413 -0.47907100 -0.001588833
## x2bar      0.57264414 -0.545334126  1.00000000  0.42422130  0.224489979
## x2ybar     0.60459376 -0.479071003  0.42422130  1.00000000  0.407195688
## y2bar      0.23752320 -0.001588833  0.22448998  0.40719569  1.000000000
## yedgexcor  0.09709965  0.093044671  0.32597229 -0.08340778  0.307083845
## xy2bar    -0.65426153  0.415388547 -0.20107898 -0.44653082 -0.178347483
## xedgeycor  0.63222470 -0.460015125  0.28843650  0.60552393  0.120295037
## yedge      0.21351577 -0.121359566  0.49464839  0.48443882  0.590480306
## xbox       0.11381436  0.091956235  0.11465700  0.07827837  0.295688120
## onpix      0.07666080  0.131845511  0.04010194  0.31495728  0.349300771
## xybar      0.26516264  0.178460382 -0.13366642  0.05186707  0.060647127
## xedge     -0.15096610  0.237781249 -0.12943288  0.06462221  0.108183786
## width     -0.01395027  0.242914440 -0.12404962 -0.02755762  0.275115570
## height     0.05842910  0.086954511  0.16071778  0.04118976  0.204347683
## ybox       0.01647361  0.164771532  0.09240318 -0.03878598  0.231910249
## .rnorm    -0.02167939  0.028019172 -0.01478605 -0.03107159  0.004216096
##              yedgexcor       xy2bar    xedgeycor       yedge         xbox
## ybar       0.097099649 -0.654261532  0.632224701  0.21351577  0.113814363
## xbar       0.093044671  0.415388547 -0.460015125 -0.12135957  0.091956235
## x2bar      0.325972289 -0.201078976  0.288436498  0.49464839  0.114657003
## x2ybar    -0.083407783 -0.446530820  0.605523934  0.48443882  0.078278371
## y2bar      0.307083845 -0.178347483  0.120295037  0.59048031  0.295688120
## yedgexcor  1.000000000  0.153765326 -0.241111018  0.17551195  0.038265541
## xy2bar     0.153765326  1.000000000 -0.743214471  0.08455057 -0.001511163
## xedgeycor -0.241111018 -0.743214471  1.000000000  0.09985942  0.048382011
## yedge      0.175511951  0.084550575  0.099859420  1.00000000  0.423196740
## xbox       0.038265541 -0.001511163  0.048382011  0.42319674  1.000000000
## onpix     -0.072312872  0.052489797  0.156205863  0.58891116  0.596906428
## xybar      0.059012315 -0.503880904  0.305133045 -0.27447605  0.257187072
## xedge     -0.092864546  0.398113898 -0.157722822  0.46271162  0.551552203
## width     -0.087045979 -0.006071783  0.058712050  0.31177993  0.808700590
## height     0.014845501  0.057460765  0.085303728  0.38142332  0.683670460
## ybox       0.032152528  0.069536328 -0.021815190  0.33653578  0.795287479
## .rnorm    -0.003883444 -0.010871137  0.004739183 -0.02368955 -0.018316514
##                 onpix       xybar       xedge        width      height
## ybar       0.07666080  0.26516264 -0.15096610 -0.013950275  0.05842910
## xbar       0.13184551  0.17846038  0.23778125  0.242914440  0.08695451
## x2bar      0.04010194 -0.13366642 -0.12943288 -0.124049618  0.16071778
## x2ybar     0.31495728  0.05186707  0.06462221 -0.027557625  0.04118976
## y2bar      0.34930077  0.06064713  0.10818379  0.275115570  0.20434768
## yedgexcor -0.07231287  0.05901231 -0.09286455 -0.087045979  0.01484550
## xy2bar     0.05248980 -0.50388090  0.39811390 -0.006071783  0.05746076
## xedgeycor  0.15620586  0.30513304 -0.15772282  0.058712050  0.08530373
## yedge      0.58891116 -0.27447605  0.46271162  0.311779935  0.38142332
## xbox       0.59690643  0.25718707  0.55155220  0.808700590  0.68367046
## onpix      1.00000000 -0.09042431  0.62357011  0.756372096  0.73399057
## xybar     -0.09042431  1.00000000 -0.19141573  0.198033339  0.04897808
## xedge      0.62357011 -0.19141573  1.00000000  0.510061567  0.42197072
## width      0.75637210  0.19803334  0.51006157  1.000000000  0.76363197
## height     0.73399057  0.04897808  0.42197072  0.763631969  1.00000000
## ybox       0.59450994  0.22449073  0.42884870  0.744532920  0.84299910
## .rnorm    -0.01698361  0.01009306 -0.02131315 -0.022024671 -0.01147920
##                   ybox       .rnorm
## ybar       0.016473612 -0.021679389
## xbar       0.164771532  0.028019172
## x2bar      0.092403183 -0.014786054
## x2ybar    -0.038785976 -0.031071588
## y2bar      0.231910249  0.004216096
## yedgexcor  0.032152528 -0.003883444
## xy2bar     0.069536328 -0.010871137
## xedgeycor -0.021815190  0.004739183
## yedge      0.336535776 -0.023689553
## xbox       0.795287479 -0.018316514
## onpix      0.594509942 -0.016983612
## xybar      0.224490725  0.010093059
## xedge      0.428848697 -0.021313146
## width      0.744532920 -0.022024671
## height     0.842999095 -0.011479204
## ybox       1.000000000 -0.006360588
## .rnorm    -0.006360588  1.000000000
##                 ybar        xbar      x2bar     x2ybar       y2bar
## ybar      0.00000000 0.688602938 0.57264414 0.60459376 0.237523198
## xbar      0.68860294 0.000000000 0.54533413 0.47907100 0.001588833
## x2bar     0.57264414 0.545334126 0.00000000 0.42422130 0.224489979
## x2ybar    0.60459376 0.479071003 0.42422130 0.00000000 0.407195688
## y2bar     0.23752320 0.001588833 0.22448998 0.40719569 0.000000000
## yedgexcor 0.09709965 0.093044671 0.32597229 0.08340778 0.307083845
## xy2bar    0.65426153 0.415388547 0.20107898 0.44653082 0.178347483
## xedgeycor 0.63222470 0.460015125 0.28843650 0.60552393 0.120295037
## yedge     0.21351577 0.121359566 0.49464839 0.48443882 0.590480306
## xbox      0.11381436 0.091956235 0.11465700 0.07827837 0.295688120
## onpix     0.07666080 0.131845511 0.04010194 0.31495728 0.349300771
## xybar     0.26516264 0.178460382 0.13366642 0.05186707 0.060647127
## xedge     0.15096610 0.237781249 0.12943288 0.06462221 0.108183786
## width     0.01395027 0.242914440 0.12404962 0.02755762 0.275115570
## height    0.05842910 0.086954511 0.16071778 0.04118976 0.204347683
## ybox      0.01647361 0.164771532 0.09240318 0.03878598 0.231910249
## .rnorm    0.02167939 0.028019172 0.01478605 0.03107159 0.004216096
##             yedgexcor      xy2bar   xedgeycor      yedge        xbox
## ybar      0.097099649 0.654261532 0.632224701 0.21351577 0.113814363
## xbar      0.093044671 0.415388547 0.460015125 0.12135957 0.091956235
## x2bar     0.325972289 0.201078976 0.288436498 0.49464839 0.114657003
## x2ybar    0.083407783 0.446530820 0.605523934 0.48443882 0.078278371
## y2bar     0.307083845 0.178347483 0.120295037 0.59048031 0.295688120
## yedgexcor 0.000000000 0.153765326 0.241111018 0.17551195 0.038265541
## xy2bar    0.153765326 0.000000000 0.743214471 0.08455057 0.001511163
## xedgeycor 0.241111018 0.743214471 0.000000000 0.09985942 0.048382011
## yedge     0.175511951 0.084550575 0.099859420 0.00000000 0.423196740
## xbox      0.038265541 0.001511163 0.048382011 0.42319674 0.000000000
## onpix     0.072312872 0.052489797 0.156205863 0.58891116 0.596906428
## xybar     0.059012315 0.503880904 0.305133045 0.27447605 0.257187072
## xedge     0.092864546 0.398113898 0.157722822 0.46271162 0.551552203
## width     0.087045979 0.006071783 0.058712050 0.31177993 0.808700590
## height    0.014845501 0.057460765 0.085303728 0.38142332 0.683670460
## ybox      0.032152528 0.069536328 0.021815190 0.33653578 0.795287479
## .rnorm    0.003883444 0.010871137 0.004739183 0.02368955 0.018316514
##                onpix      xybar      xedge       width     height
## ybar      0.07666080 0.26516264 0.15096610 0.013950275 0.05842910
## xbar      0.13184551 0.17846038 0.23778125 0.242914440 0.08695451
## x2bar     0.04010194 0.13366642 0.12943288 0.124049618 0.16071778
## x2ybar    0.31495728 0.05186707 0.06462221 0.027557625 0.04118976
## y2bar     0.34930077 0.06064713 0.10818379 0.275115570 0.20434768
## yedgexcor 0.07231287 0.05901231 0.09286455 0.087045979 0.01484550
## xy2bar    0.05248980 0.50388090 0.39811390 0.006071783 0.05746076
## xedgeycor 0.15620586 0.30513304 0.15772282 0.058712050 0.08530373
## yedge     0.58891116 0.27447605 0.46271162 0.311779935 0.38142332
## xbox      0.59690643 0.25718707 0.55155220 0.808700590 0.68367046
## onpix     0.00000000 0.09042431 0.62357011 0.756372096 0.73399057
## xybar     0.09042431 0.00000000 0.19141573 0.198033339 0.04897808
## xedge     0.62357011 0.19141573 0.00000000 0.510061567 0.42197072
## width     0.75637210 0.19803334 0.51006157 0.000000000 0.76363197
## height    0.73399057 0.04897808 0.42197072 0.763631969 0.00000000
## ybox      0.59450994 0.22449073 0.42884870 0.744532920 0.84299910
## .rnorm    0.01698361 0.01009306 0.02131315 0.022024671 0.01147920
##                  ybox      .rnorm
## ybar      0.016473612 0.021679389
## xbar      0.164771532 0.028019172
## x2bar     0.092403183 0.014786054
## x2ybar    0.038785976 0.031071588
## y2bar     0.231910249 0.004216096
## yedgexcor 0.032152528 0.003883444
## xy2bar    0.069536328 0.010871137
## xedgeycor 0.021815190 0.004739183
## yedge     0.336535776 0.023689553
## xbox      0.795287479 0.018316514
## onpix     0.594509942 0.016983612
## xybar     0.224490725 0.010093059
## xedge     0.428848697 0.021313146
## width     0.744532920 0.022024671
## height    0.842999095 0.011479204
## ybox      0.000000000 0.006360588
## .rnorm    0.006360588 0.000000000
## [1] "cor(height, ybox)=0.8430"
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-1.png) 

```
## [1] "cor(letter.fctr, height)=0.0457"
## [1] "cor(letter.fctr, ybox)=0.0369"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified ybox as highly correlated with other features
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-2.png) 

```
## [1] "checking correlations for features:"
##  [1] "ybar"      "xbar"      "x2bar"     "x2ybar"    "y2bar"    
##  [6] "yedgexcor" "xy2bar"    "xedgeycor" "yedge"     "xbox"     
## [11] "onpix"     "xybar"     "xedge"     "width"     "height"   
## [16] ".rnorm"   
##                  ybar         xbar       x2bar      x2ybar        y2bar
## ybar       1.00000000 -0.688602938  0.57264414  0.60459376  0.237523198
## xbar      -0.68860294  1.000000000 -0.54533413 -0.47907100 -0.001588833
## x2bar      0.57264414 -0.545334126  1.00000000  0.42422130  0.224489979
## x2ybar     0.60459376 -0.479071003  0.42422130  1.00000000  0.407195688
## y2bar      0.23752320 -0.001588833  0.22448998  0.40719569  1.000000000
## yedgexcor  0.09709965  0.093044671  0.32597229 -0.08340778  0.307083845
## xy2bar    -0.65426153  0.415388547 -0.20107898 -0.44653082 -0.178347483
## xedgeycor  0.63222470 -0.460015125  0.28843650  0.60552393  0.120295037
## yedge      0.21351577 -0.121359566  0.49464839  0.48443882  0.590480306
## xbox       0.11381436  0.091956235  0.11465700  0.07827837  0.295688120
## onpix      0.07666080  0.131845511  0.04010194  0.31495728  0.349300771
## xybar      0.26516264  0.178460382 -0.13366642  0.05186707  0.060647127
## xedge     -0.15096610  0.237781249 -0.12943288  0.06462221  0.108183786
## width     -0.01395027  0.242914440 -0.12404962 -0.02755762  0.275115570
## height     0.05842910  0.086954511  0.16071778  0.04118976  0.204347683
## .rnorm    -0.02167939  0.028019172 -0.01478605 -0.03107159  0.004216096
##              yedgexcor       xy2bar    xedgeycor       yedge         xbox
## ybar       0.097099649 -0.654261532  0.632224701  0.21351577  0.113814363
## xbar       0.093044671  0.415388547 -0.460015125 -0.12135957  0.091956235
## x2bar      0.325972289 -0.201078976  0.288436498  0.49464839  0.114657003
## x2ybar    -0.083407783 -0.446530820  0.605523934  0.48443882  0.078278371
## y2bar      0.307083845 -0.178347483  0.120295037  0.59048031  0.295688120
## yedgexcor  1.000000000  0.153765326 -0.241111018  0.17551195  0.038265541
## xy2bar     0.153765326  1.000000000 -0.743214471  0.08455057 -0.001511163
## xedgeycor -0.241111018 -0.743214471  1.000000000  0.09985942  0.048382011
## yedge      0.175511951  0.084550575  0.099859420  1.00000000  0.423196740
## xbox       0.038265541 -0.001511163  0.048382011  0.42319674  1.000000000
## onpix     -0.072312872  0.052489797  0.156205863  0.58891116  0.596906428
## xybar      0.059012315 -0.503880904  0.305133045 -0.27447605  0.257187072
## xedge     -0.092864546  0.398113898 -0.157722822  0.46271162  0.551552203
## width     -0.087045979 -0.006071783  0.058712050  0.31177993  0.808700590
## height     0.014845501  0.057460765  0.085303728  0.38142332  0.683670460
## .rnorm    -0.003883444 -0.010871137  0.004739183 -0.02368955 -0.018316514
##                 onpix       xybar       xedge        width      height
## ybar       0.07666080  0.26516264 -0.15096610 -0.013950275  0.05842910
## xbar       0.13184551  0.17846038  0.23778125  0.242914440  0.08695451
## x2bar      0.04010194 -0.13366642 -0.12943288 -0.124049618  0.16071778
## x2ybar     0.31495728  0.05186707  0.06462221 -0.027557625  0.04118976
## y2bar      0.34930077  0.06064713  0.10818379  0.275115570  0.20434768
## yedgexcor -0.07231287  0.05901231 -0.09286455 -0.087045979  0.01484550
## xy2bar     0.05248980 -0.50388090  0.39811390 -0.006071783  0.05746076
## xedgeycor  0.15620586  0.30513304 -0.15772282  0.058712050  0.08530373
## yedge      0.58891116 -0.27447605  0.46271162  0.311779935  0.38142332
## xbox       0.59690643  0.25718707  0.55155220  0.808700590  0.68367046
## onpix      1.00000000 -0.09042431  0.62357011  0.756372096  0.73399057
## xybar     -0.09042431  1.00000000 -0.19141573  0.198033339  0.04897808
## xedge      0.62357011 -0.19141573  1.00000000  0.510061567  0.42197072
## width      0.75637210  0.19803334  0.51006157  1.000000000  0.76363197
## height     0.73399057  0.04897808  0.42197072  0.763631969  1.00000000
## .rnorm    -0.01698361  0.01009306 -0.02131315 -0.022024671 -0.01147920
##                 .rnorm
## ybar      -0.021679389
## xbar       0.028019172
## x2bar     -0.014786054
## x2ybar    -0.031071588
## y2bar      0.004216096
## yedgexcor -0.003883444
## xy2bar    -0.010871137
## xedgeycor  0.004739183
## yedge     -0.023689553
## xbox      -0.018316514
## onpix     -0.016983612
## xybar      0.010093059
## xedge     -0.021313146
## width     -0.022024671
## height    -0.011479204
## .rnorm     1.000000000
##                 ybar        xbar      x2bar     x2ybar       y2bar
## ybar      0.00000000 0.688602938 0.57264414 0.60459376 0.237523198
## xbar      0.68860294 0.000000000 0.54533413 0.47907100 0.001588833
## x2bar     0.57264414 0.545334126 0.00000000 0.42422130 0.224489979
## x2ybar    0.60459376 0.479071003 0.42422130 0.00000000 0.407195688
## y2bar     0.23752320 0.001588833 0.22448998 0.40719569 0.000000000
## yedgexcor 0.09709965 0.093044671 0.32597229 0.08340778 0.307083845
## xy2bar    0.65426153 0.415388547 0.20107898 0.44653082 0.178347483
## xedgeycor 0.63222470 0.460015125 0.28843650 0.60552393 0.120295037
## yedge     0.21351577 0.121359566 0.49464839 0.48443882 0.590480306
## xbox      0.11381436 0.091956235 0.11465700 0.07827837 0.295688120
## onpix     0.07666080 0.131845511 0.04010194 0.31495728 0.349300771
## xybar     0.26516264 0.178460382 0.13366642 0.05186707 0.060647127
## xedge     0.15096610 0.237781249 0.12943288 0.06462221 0.108183786
## width     0.01395027 0.242914440 0.12404962 0.02755762 0.275115570
## height    0.05842910 0.086954511 0.16071778 0.04118976 0.204347683
## .rnorm    0.02167939 0.028019172 0.01478605 0.03107159 0.004216096
##             yedgexcor      xy2bar   xedgeycor      yedge        xbox
## ybar      0.097099649 0.654261532 0.632224701 0.21351577 0.113814363
## xbar      0.093044671 0.415388547 0.460015125 0.12135957 0.091956235
## x2bar     0.325972289 0.201078976 0.288436498 0.49464839 0.114657003
## x2ybar    0.083407783 0.446530820 0.605523934 0.48443882 0.078278371
## y2bar     0.307083845 0.178347483 0.120295037 0.59048031 0.295688120
## yedgexcor 0.000000000 0.153765326 0.241111018 0.17551195 0.038265541
## xy2bar    0.153765326 0.000000000 0.743214471 0.08455057 0.001511163
## xedgeycor 0.241111018 0.743214471 0.000000000 0.09985942 0.048382011
## yedge     0.175511951 0.084550575 0.099859420 0.00000000 0.423196740
## xbox      0.038265541 0.001511163 0.048382011 0.42319674 0.000000000
## onpix     0.072312872 0.052489797 0.156205863 0.58891116 0.596906428
## xybar     0.059012315 0.503880904 0.305133045 0.27447605 0.257187072
## xedge     0.092864546 0.398113898 0.157722822 0.46271162 0.551552203
## width     0.087045979 0.006071783 0.058712050 0.31177993 0.808700590
## height    0.014845501 0.057460765 0.085303728 0.38142332 0.683670460
## .rnorm    0.003883444 0.010871137 0.004739183 0.02368955 0.018316514
##                onpix      xybar      xedge       width     height
## ybar      0.07666080 0.26516264 0.15096610 0.013950275 0.05842910
## xbar      0.13184551 0.17846038 0.23778125 0.242914440 0.08695451
## x2bar     0.04010194 0.13366642 0.12943288 0.124049618 0.16071778
## x2ybar    0.31495728 0.05186707 0.06462221 0.027557625 0.04118976
## y2bar     0.34930077 0.06064713 0.10818379 0.275115570 0.20434768
## yedgexcor 0.07231287 0.05901231 0.09286455 0.087045979 0.01484550
## xy2bar    0.05248980 0.50388090 0.39811390 0.006071783 0.05746076
## xedgeycor 0.15620586 0.30513304 0.15772282 0.058712050 0.08530373
## yedge     0.58891116 0.27447605 0.46271162 0.311779935 0.38142332
## xbox      0.59690643 0.25718707 0.55155220 0.808700590 0.68367046
## onpix     0.00000000 0.09042431 0.62357011 0.756372096 0.73399057
## xybar     0.09042431 0.00000000 0.19141573 0.198033339 0.04897808
## xedge     0.62357011 0.19141573 0.00000000 0.510061567 0.42197072
## width     0.75637210 0.19803334 0.51006157 0.000000000 0.76363197
## height    0.73399057 0.04897808 0.42197072 0.763631969 0.00000000
## .rnorm    0.01698361 0.01009306 0.02131315 0.022024671 0.01147920
##                .rnorm
## ybar      0.021679389
## xbar      0.028019172
## x2bar     0.014786054
## x2ybar    0.031071588
## y2bar     0.004216096
## yedgexcor 0.003883444
## xy2bar    0.010871137
## xedgeycor 0.004739183
## yedge     0.023689553
## xbox      0.018316514
## onpix     0.016983612
## xybar     0.010093059
## xedge     0.021313146
## width     0.022024671
## height    0.011479204
## .rnorm    0.000000000
## [1] "cor(xbox, width)=0.8087"
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-3.png) 

```
## [1] "cor(letter.fctr, xbox)=0.1683"
## [1] "cor(letter.fctr, width)=0.0491"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified width as highly correlated with other features
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-4.png) 

```
## [1] "checking correlations for features:"
##  [1] "ybar"      "xbar"      "x2bar"     "x2ybar"    "y2bar"    
##  [6] "yedgexcor" "xy2bar"    "xedgeycor" "yedge"     "xbox"     
## [11] "onpix"     "xybar"     "xedge"     "height"    ".rnorm"   
##                  ybar         xbar       x2bar      x2ybar        y2bar
## ybar       1.00000000 -0.688602938  0.57264414  0.60459376  0.237523198
## xbar      -0.68860294  1.000000000 -0.54533413 -0.47907100 -0.001588833
## x2bar      0.57264414 -0.545334126  1.00000000  0.42422130  0.224489979
## x2ybar     0.60459376 -0.479071003  0.42422130  1.00000000  0.407195688
## y2bar      0.23752320 -0.001588833  0.22448998  0.40719569  1.000000000
## yedgexcor  0.09709965  0.093044671  0.32597229 -0.08340778  0.307083845
## xy2bar    -0.65426153  0.415388547 -0.20107898 -0.44653082 -0.178347483
## xedgeycor  0.63222470 -0.460015125  0.28843650  0.60552393  0.120295037
## yedge      0.21351577 -0.121359566  0.49464839  0.48443882  0.590480306
## xbox       0.11381436  0.091956235  0.11465700  0.07827837  0.295688120
## onpix      0.07666080  0.131845511  0.04010194  0.31495728  0.349300771
## xybar      0.26516264  0.178460382 -0.13366642  0.05186707  0.060647127
## xedge     -0.15096610  0.237781249 -0.12943288  0.06462221  0.108183786
## height     0.05842910  0.086954511  0.16071778  0.04118976  0.204347683
## .rnorm    -0.02167939  0.028019172 -0.01478605 -0.03107159  0.004216096
##              yedgexcor       xy2bar    xedgeycor       yedge         xbox
## ybar       0.097099649 -0.654261532  0.632224701  0.21351577  0.113814363
## xbar       0.093044671  0.415388547 -0.460015125 -0.12135957  0.091956235
## x2bar      0.325972289 -0.201078976  0.288436498  0.49464839  0.114657003
## x2ybar    -0.083407783 -0.446530820  0.605523934  0.48443882  0.078278371
## y2bar      0.307083845 -0.178347483  0.120295037  0.59048031  0.295688120
## yedgexcor  1.000000000  0.153765326 -0.241111018  0.17551195  0.038265541
## xy2bar     0.153765326  1.000000000 -0.743214471  0.08455057 -0.001511163
## xedgeycor -0.241111018 -0.743214471  1.000000000  0.09985942  0.048382011
## yedge      0.175511951  0.084550575  0.099859420  1.00000000  0.423196740
## xbox       0.038265541 -0.001511163  0.048382011  0.42319674  1.000000000
## onpix     -0.072312872  0.052489797  0.156205863  0.58891116  0.596906428
## xybar      0.059012315 -0.503880904  0.305133045 -0.27447605  0.257187072
## xedge     -0.092864546  0.398113898 -0.157722822  0.46271162  0.551552203
## height     0.014845501  0.057460765  0.085303728  0.38142332  0.683670460
## .rnorm    -0.003883444 -0.010871137  0.004739183 -0.02368955 -0.018316514
##                 onpix       xybar       xedge      height       .rnorm
## ybar       0.07666080  0.26516264 -0.15096610  0.05842910 -0.021679389
## xbar       0.13184551  0.17846038  0.23778125  0.08695451  0.028019172
## x2bar      0.04010194 -0.13366642 -0.12943288  0.16071778 -0.014786054
## x2ybar     0.31495728  0.05186707  0.06462221  0.04118976 -0.031071588
## y2bar      0.34930077  0.06064713  0.10818379  0.20434768  0.004216096
## yedgexcor -0.07231287  0.05901231 -0.09286455  0.01484550 -0.003883444
## xy2bar     0.05248980 -0.50388090  0.39811390  0.05746076 -0.010871137
## xedgeycor  0.15620586  0.30513304 -0.15772282  0.08530373  0.004739183
## yedge      0.58891116 -0.27447605  0.46271162  0.38142332 -0.023689553
## xbox       0.59690643  0.25718707  0.55155220  0.68367046 -0.018316514
## onpix      1.00000000 -0.09042431  0.62357011  0.73399057 -0.016983612
## xybar     -0.09042431  1.00000000 -0.19141573  0.04897808  0.010093059
## xedge      0.62357011 -0.19141573  1.00000000  0.42197072 -0.021313146
## height     0.73399057  0.04897808  0.42197072  1.00000000 -0.011479204
## .rnorm    -0.01698361  0.01009306 -0.02131315 -0.01147920  1.000000000
##                 ybar        xbar      x2bar     x2ybar       y2bar
## ybar      0.00000000 0.688602938 0.57264414 0.60459376 0.237523198
## xbar      0.68860294 0.000000000 0.54533413 0.47907100 0.001588833
## x2bar     0.57264414 0.545334126 0.00000000 0.42422130 0.224489979
## x2ybar    0.60459376 0.479071003 0.42422130 0.00000000 0.407195688
## y2bar     0.23752320 0.001588833 0.22448998 0.40719569 0.000000000
## yedgexcor 0.09709965 0.093044671 0.32597229 0.08340778 0.307083845
## xy2bar    0.65426153 0.415388547 0.20107898 0.44653082 0.178347483
## xedgeycor 0.63222470 0.460015125 0.28843650 0.60552393 0.120295037
## yedge     0.21351577 0.121359566 0.49464839 0.48443882 0.590480306
## xbox      0.11381436 0.091956235 0.11465700 0.07827837 0.295688120
## onpix     0.07666080 0.131845511 0.04010194 0.31495728 0.349300771
## xybar     0.26516264 0.178460382 0.13366642 0.05186707 0.060647127
## xedge     0.15096610 0.237781249 0.12943288 0.06462221 0.108183786
## height    0.05842910 0.086954511 0.16071778 0.04118976 0.204347683
## .rnorm    0.02167939 0.028019172 0.01478605 0.03107159 0.004216096
##             yedgexcor      xy2bar   xedgeycor      yedge        xbox
## ybar      0.097099649 0.654261532 0.632224701 0.21351577 0.113814363
## xbar      0.093044671 0.415388547 0.460015125 0.12135957 0.091956235
## x2bar     0.325972289 0.201078976 0.288436498 0.49464839 0.114657003
## x2ybar    0.083407783 0.446530820 0.605523934 0.48443882 0.078278371
## y2bar     0.307083845 0.178347483 0.120295037 0.59048031 0.295688120
## yedgexcor 0.000000000 0.153765326 0.241111018 0.17551195 0.038265541
## xy2bar    0.153765326 0.000000000 0.743214471 0.08455057 0.001511163
## xedgeycor 0.241111018 0.743214471 0.000000000 0.09985942 0.048382011
## yedge     0.175511951 0.084550575 0.099859420 0.00000000 0.423196740
## xbox      0.038265541 0.001511163 0.048382011 0.42319674 0.000000000
## onpix     0.072312872 0.052489797 0.156205863 0.58891116 0.596906428
## xybar     0.059012315 0.503880904 0.305133045 0.27447605 0.257187072
## xedge     0.092864546 0.398113898 0.157722822 0.46271162 0.551552203
## height    0.014845501 0.057460765 0.085303728 0.38142332 0.683670460
## .rnorm    0.003883444 0.010871137 0.004739183 0.02368955 0.018316514
##                onpix      xybar      xedge     height      .rnorm
## ybar      0.07666080 0.26516264 0.15096610 0.05842910 0.021679389
## xbar      0.13184551 0.17846038 0.23778125 0.08695451 0.028019172
## x2bar     0.04010194 0.13366642 0.12943288 0.16071778 0.014786054
## x2ybar    0.31495728 0.05186707 0.06462221 0.04118976 0.031071588
## y2bar     0.34930077 0.06064713 0.10818379 0.20434768 0.004216096
## yedgexcor 0.07231287 0.05901231 0.09286455 0.01484550 0.003883444
## xy2bar    0.05248980 0.50388090 0.39811390 0.05746076 0.010871137
## xedgeycor 0.15620586 0.30513304 0.15772282 0.08530373 0.004739183
## yedge     0.58891116 0.27447605 0.46271162 0.38142332 0.023689553
## xbox      0.59690643 0.25718707 0.55155220 0.68367046 0.018316514
## onpix     0.00000000 0.09042431 0.62357011 0.73399057 0.016983612
## xybar     0.09042431 0.00000000 0.19141573 0.04897808 0.010093059
## xedge     0.62357011 0.19141573 0.00000000 0.42197072 0.021313146
## height    0.73399057 0.04897808 0.42197072 0.00000000 0.011479204
## .rnorm    0.01698361 0.01009306 0.02131315 0.01147920 0.000000000
## [1] "cor(xy2bar, xedgeycor)=-0.7432"
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-5.png) 

```
## [1] "cor(letter.fctr, xy2bar)=-0.2796"
## [1] "cor(letter.fctr, xedgeycor)=0.2745"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified xedgeycor as highly correlated with other
## features
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-6.png) 

```
## [1] "checking correlations for features:"
##  [1] "ybar"      "xbar"      "x2bar"     "x2ybar"    "y2bar"    
##  [6] "yedgexcor" "xy2bar"    "yedge"     "xbox"      "onpix"    
## [11] "xybar"     "xedge"     "height"    ".rnorm"   
##                  ybar         xbar       x2bar      x2ybar        y2bar
## ybar       1.00000000 -0.688602938  0.57264414  0.60459376  0.237523198
## xbar      -0.68860294  1.000000000 -0.54533413 -0.47907100 -0.001588833
## x2bar      0.57264414 -0.545334126  1.00000000  0.42422130  0.224489979
## x2ybar     0.60459376 -0.479071003  0.42422130  1.00000000  0.407195688
## y2bar      0.23752320 -0.001588833  0.22448998  0.40719569  1.000000000
## yedgexcor  0.09709965  0.093044671  0.32597229 -0.08340778  0.307083845
## xy2bar    -0.65426153  0.415388547 -0.20107898 -0.44653082 -0.178347483
## yedge      0.21351577 -0.121359566  0.49464839  0.48443882  0.590480306
## xbox       0.11381436  0.091956235  0.11465700  0.07827837  0.295688120
## onpix      0.07666080  0.131845511  0.04010194  0.31495728  0.349300771
## xybar      0.26516264  0.178460382 -0.13366642  0.05186707  0.060647127
## xedge     -0.15096610  0.237781249 -0.12943288  0.06462221  0.108183786
## height     0.05842910  0.086954511  0.16071778  0.04118976  0.204347683
## .rnorm    -0.02167939  0.028019172 -0.01478605 -0.03107159  0.004216096
##              yedgexcor       xy2bar       yedge         xbox       onpix
## ybar       0.097099649 -0.654261532  0.21351577  0.113814363  0.07666080
## xbar       0.093044671  0.415388547 -0.12135957  0.091956235  0.13184551
## x2bar      0.325972289 -0.201078976  0.49464839  0.114657003  0.04010194
## x2ybar    -0.083407783 -0.446530820  0.48443882  0.078278371  0.31495728
## y2bar      0.307083845 -0.178347483  0.59048031  0.295688120  0.34930077
## yedgexcor  1.000000000  0.153765326  0.17551195  0.038265541 -0.07231287
## xy2bar     0.153765326  1.000000000  0.08455057 -0.001511163  0.05248980
## yedge      0.175511951  0.084550575  1.00000000  0.423196740  0.58891116
## xbox       0.038265541 -0.001511163  0.42319674  1.000000000  0.59690643
## onpix     -0.072312872  0.052489797  0.58891116  0.596906428  1.00000000
## xybar      0.059012315 -0.503880904 -0.27447605  0.257187072 -0.09042431
## xedge     -0.092864546  0.398113898  0.46271162  0.551552203  0.62357011
## height     0.014845501  0.057460765  0.38142332  0.683670460  0.73399057
## .rnorm    -0.003883444 -0.010871137 -0.02368955 -0.018316514 -0.01698361
##                 xybar       xedge      height       .rnorm
## ybar       0.26516264 -0.15096610  0.05842910 -0.021679389
## xbar       0.17846038  0.23778125  0.08695451  0.028019172
## x2bar     -0.13366642 -0.12943288  0.16071778 -0.014786054
## x2ybar     0.05186707  0.06462221  0.04118976 -0.031071588
## y2bar      0.06064713  0.10818379  0.20434768  0.004216096
## yedgexcor  0.05901231 -0.09286455  0.01484550 -0.003883444
## xy2bar    -0.50388090  0.39811390  0.05746076 -0.010871137
## yedge     -0.27447605  0.46271162  0.38142332 -0.023689553
## xbox       0.25718707  0.55155220  0.68367046 -0.018316514
## onpix     -0.09042431  0.62357011  0.73399057 -0.016983612
## xybar      1.00000000 -0.19141573  0.04897808  0.010093059
## xedge     -0.19141573  1.00000000  0.42197072 -0.021313146
## height     0.04897808  0.42197072  1.00000000 -0.011479204
## .rnorm     0.01009306 -0.02131315 -0.01147920  1.000000000
##                 ybar        xbar      x2bar     x2ybar       y2bar
## ybar      0.00000000 0.688602938 0.57264414 0.60459376 0.237523198
## xbar      0.68860294 0.000000000 0.54533413 0.47907100 0.001588833
## x2bar     0.57264414 0.545334126 0.00000000 0.42422130 0.224489979
## x2ybar    0.60459376 0.479071003 0.42422130 0.00000000 0.407195688
## y2bar     0.23752320 0.001588833 0.22448998 0.40719569 0.000000000
## yedgexcor 0.09709965 0.093044671 0.32597229 0.08340778 0.307083845
## xy2bar    0.65426153 0.415388547 0.20107898 0.44653082 0.178347483
## yedge     0.21351577 0.121359566 0.49464839 0.48443882 0.590480306
## xbox      0.11381436 0.091956235 0.11465700 0.07827837 0.295688120
## onpix     0.07666080 0.131845511 0.04010194 0.31495728 0.349300771
## xybar     0.26516264 0.178460382 0.13366642 0.05186707 0.060647127
## xedge     0.15096610 0.237781249 0.12943288 0.06462221 0.108183786
## height    0.05842910 0.086954511 0.16071778 0.04118976 0.204347683
## .rnorm    0.02167939 0.028019172 0.01478605 0.03107159 0.004216096
##             yedgexcor      xy2bar      yedge        xbox      onpix
## ybar      0.097099649 0.654261532 0.21351577 0.113814363 0.07666080
## xbar      0.093044671 0.415388547 0.12135957 0.091956235 0.13184551
## x2bar     0.325972289 0.201078976 0.49464839 0.114657003 0.04010194
## x2ybar    0.083407783 0.446530820 0.48443882 0.078278371 0.31495728
## y2bar     0.307083845 0.178347483 0.59048031 0.295688120 0.34930077
## yedgexcor 0.000000000 0.153765326 0.17551195 0.038265541 0.07231287
## xy2bar    0.153765326 0.000000000 0.08455057 0.001511163 0.05248980
## yedge     0.175511951 0.084550575 0.00000000 0.423196740 0.58891116
## xbox      0.038265541 0.001511163 0.42319674 0.000000000 0.59690643
## onpix     0.072312872 0.052489797 0.58891116 0.596906428 0.00000000
## xybar     0.059012315 0.503880904 0.27447605 0.257187072 0.09042431
## xedge     0.092864546 0.398113898 0.46271162 0.551552203 0.62357011
## height    0.014845501 0.057460765 0.38142332 0.683670460 0.73399057
## .rnorm    0.003883444 0.010871137 0.02368955 0.018316514 0.01698361
##                xybar      xedge     height      .rnorm
## ybar      0.26516264 0.15096610 0.05842910 0.021679389
## xbar      0.17846038 0.23778125 0.08695451 0.028019172
## x2bar     0.13366642 0.12943288 0.16071778 0.014786054
## x2ybar    0.05186707 0.06462221 0.04118976 0.031071588
## y2bar     0.06064713 0.10818379 0.20434768 0.004216096
## yedgexcor 0.05901231 0.09286455 0.01484550 0.003883444
## xy2bar    0.50388090 0.39811390 0.05746076 0.010871137
## yedge     0.27447605 0.46271162 0.38142332 0.023689553
## xbox      0.25718707 0.55155220 0.68367046 0.018316514
## onpix     0.09042431 0.62357011 0.73399057 0.016983612
## xybar     0.00000000 0.19141573 0.04897808 0.010093059
## xedge     0.19141573 0.00000000 0.42197072 0.021313146
## height    0.04897808 0.42197072 0.00000000 0.011479204
## .rnorm    0.01009306 0.02131315 0.01147920 0.000000000
## [1] "cor(onpix, height)=0.7340"
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-7.png) 

```
## [1] "cor(letter.fctr, onpix)=0.1672"
## [1] "cor(letter.fctr, height)=0.0457"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified height as highly correlated with other
## features
```

![](Letter_Recognition_files/figure-html/remove_correlated_features-8.png) 

```
## [1] "checking correlations for features:"
##  [1] "ybar"      "xbar"      "x2bar"     "x2ybar"    "y2bar"    
##  [6] "yedgexcor" "xy2bar"    "yedge"     "xbox"      "onpix"    
## [11] "xybar"     "xedge"     ".rnorm"   
##                  ybar         xbar       x2bar      x2ybar        y2bar
## ybar       1.00000000 -0.688602938  0.57264414  0.60459376  0.237523198
## xbar      -0.68860294  1.000000000 -0.54533413 -0.47907100 -0.001588833
## x2bar      0.57264414 -0.545334126  1.00000000  0.42422130  0.224489979
## x2ybar     0.60459376 -0.479071003  0.42422130  1.00000000  0.407195688
## y2bar      0.23752320 -0.001588833  0.22448998  0.40719569  1.000000000
## yedgexcor  0.09709965  0.093044671  0.32597229 -0.08340778  0.307083845
## xy2bar    -0.65426153  0.415388547 -0.20107898 -0.44653082 -0.178347483
## yedge      0.21351577 -0.121359566  0.49464839  0.48443882  0.590480306
## xbox       0.11381436  0.091956235  0.11465700  0.07827837  0.295688120
## onpix      0.07666080  0.131845511  0.04010194  0.31495728  0.349300771
## xybar      0.26516264  0.178460382 -0.13366642  0.05186707  0.060647127
## xedge     -0.15096610  0.237781249 -0.12943288  0.06462221  0.108183786
## .rnorm    -0.02167939  0.028019172 -0.01478605 -0.03107159  0.004216096
##              yedgexcor       xy2bar       yedge         xbox       onpix
## ybar       0.097099649 -0.654261532  0.21351577  0.113814363  0.07666080
## xbar       0.093044671  0.415388547 -0.12135957  0.091956235  0.13184551
## x2bar      0.325972289 -0.201078976  0.49464839  0.114657003  0.04010194
## x2ybar    -0.083407783 -0.446530820  0.48443882  0.078278371  0.31495728
## y2bar      0.307083845 -0.178347483  0.59048031  0.295688120  0.34930077
## yedgexcor  1.000000000  0.153765326  0.17551195  0.038265541 -0.07231287
## xy2bar     0.153765326  1.000000000  0.08455057 -0.001511163  0.05248980
## yedge      0.175511951  0.084550575  1.00000000  0.423196740  0.58891116
## xbox       0.038265541 -0.001511163  0.42319674  1.000000000  0.59690643
## onpix     -0.072312872  0.052489797  0.58891116  0.596906428  1.00000000
## xybar      0.059012315 -0.503880904 -0.27447605  0.257187072 -0.09042431
## xedge     -0.092864546  0.398113898  0.46271162  0.551552203  0.62357011
## .rnorm    -0.003883444 -0.010871137 -0.02368955 -0.018316514 -0.01698361
##                 xybar       xedge       .rnorm
## ybar       0.26516264 -0.15096610 -0.021679389
## xbar       0.17846038  0.23778125  0.028019172
## x2bar     -0.13366642 -0.12943288 -0.014786054
## x2ybar     0.05186707  0.06462221 -0.031071588
## y2bar      0.06064713  0.10818379  0.004216096
## yedgexcor  0.05901231 -0.09286455 -0.003883444
## xy2bar    -0.50388090  0.39811390 -0.010871137
## yedge     -0.27447605  0.46271162 -0.023689553
## xbox       0.25718707  0.55155220 -0.018316514
## onpix     -0.09042431  0.62357011 -0.016983612
## xybar      1.00000000 -0.19141573  0.010093059
## xedge     -0.19141573  1.00000000 -0.021313146
## .rnorm     0.01009306 -0.02131315  1.000000000
##                 ybar        xbar      x2bar     x2ybar       y2bar
## ybar      0.00000000 0.688602938 0.57264414 0.60459376 0.237523198
## xbar      0.68860294 0.000000000 0.54533413 0.47907100 0.001588833
## x2bar     0.57264414 0.545334126 0.00000000 0.42422130 0.224489979
## x2ybar    0.60459376 0.479071003 0.42422130 0.00000000 0.407195688
## y2bar     0.23752320 0.001588833 0.22448998 0.40719569 0.000000000
## yedgexcor 0.09709965 0.093044671 0.32597229 0.08340778 0.307083845
## xy2bar    0.65426153 0.415388547 0.20107898 0.44653082 0.178347483
## yedge     0.21351577 0.121359566 0.49464839 0.48443882 0.590480306
## xbox      0.11381436 0.091956235 0.11465700 0.07827837 0.295688120
## onpix     0.07666080 0.131845511 0.04010194 0.31495728 0.349300771
## xybar     0.26516264 0.178460382 0.13366642 0.05186707 0.060647127
## xedge     0.15096610 0.237781249 0.12943288 0.06462221 0.108183786
## .rnorm    0.02167939 0.028019172 0.01478605 0.03107159 0.004216096
##             yedgexcor      xy2bar      yedge        xbox      onpix
## ybar      0.097099649 0.654261532 0.21351577 0.113814363 0.07666080
## xbar      0.093044671 0.415388547 0.12135957 0.091956235 0.13184551
## x2bar     0.325972289 0.201078976 0.49464839 0.114657003 0.04010194
## x2ybar    0.083407783 0.446530820 0.48443882 0.078278371 0.31495728
## y2bar     0.307083845 0.178347483 0.59048031 0.295688120 0.34930077
## yedgexcor 0.000000000 0.153765326 0.17551195 0.038265541 0.07231287
## xy2bar    0.153765326 0.000000000 0.08455057 0.001511163 0.05248980
## yedge     0.175511951 0.084550575 0.00000000 0.423196740 0.58891116
## xbox      0.038265541 0.001511163 0.42319674 0.000000000 0.59690643
## onpix     0.072312872 0.052489797 0.58891116 0.596906428 0.00000000
## xybar     0.059012315 0.503880904 0.27447605 0.257187072 0.09042431
## xedge     0.092864546 0.398113898 0.46271162 0.551552203 0.62357011
## .rnorm    0.003883444 0.010871137 0.02368955 0.018316514 0.01698361
##                xybar      xedge      .rnorm
## ybar      0.26516264 0.15096610 0.021679389
## xbar      0.17846038 0.23778125 0.028019172
## x2bar     0.13366642 0.12943288 0.014786054
## x2ybar    0.05186707 0.06462221 0.031071588
## y2bar     0.06064713 0.10818379 0.004216096
## yedgexcor 0.05901231 0.09286455 0.003883444
## xy2bar    0.50388090 0.39811390 0.010871137
## yedge     0.27447605 0.46271162 0.023689553
## xbox      0.25718707 0.55155220 0.018316514
## onpix     0.09042431 0.62357011 0.016983612
## xybar     0.00000000 0.19141573 0.010093059
## xedge     0.19141573 0.00000000 0.021313146
## .rnorm    0.01009306 0.02131315 0.000000000
##                  id       cor.y exclude.as.feat  cor.y.abs cor.low
## ybar           ybar  0.67198759               0 0.67198759       1
## x2bar         x2bar  0.41019605               0 0.41019605       1
## x2ybar       x2ybar  0.38607514               0 0.38607514       1
## y2bar         y2bar  0.33859131               0 0.33859131       1
## yedgexcor yedgexcor  0.31367199               0 0.31367199       1
## xedgeycor xedgeycor  0.27453618               0 0.27453618       0
## yedge         yedge  0.24976884               0 0.24976884       1
## xbox           xbox  0.16830409               0 0.16830409       1
## onpix         onpix  0.16721154               0 0.16721154       1
## xybar         xybar  0.12073750               0 0.12073750       1
## xedge         xedge  0.11786463               0 0.11786463       1
## width         width  0.04909820               0 0.04909820       0
## height       height  0.04565534               0 0.04565534       0
## ybox           ybox  0.03690669               0 0.03690669       0
## .rnorm       .rnorm -0.03047401               0 0.03047401       1
## xy2bar       xy2bar -0.27957030               0 0.27957030       1
## xbar           xbar -0.41476375               0 0.41476375       1
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed7 remove_correlated_features                4                1
## elapsed8                 fit.models                5                0
##          elapsed
## elapsed7   6.727
## elapsed8  10.339
```

## Step `5`: fit models

```r
max_cor_y_x_var <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & (cor.low == 1)))[1, "id"]
if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_var != glb_Baseline_mdl_var) & 
        (glb_feats_df[max_cor_y_x_var, "cor.y.abs"] > 
         glb_feats_df[glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_var, " has a lower correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Any models that have tuning parameters has "better" results with cross-validation (except rf)
#   & "different" results for different outcome metrics

# Baseline
if (!is.null(glb_Baseline_mdl_var)) {
#     lm_mdl <- lm(reformulate(glb_Baseline_mdl_var, 
#                             response="bucket2009"), data=glb_trnent_df)
#     print(summary(lm_mdl))
#     plot(lm_mdl, ask=FALSE)
#     ret_lst <- myfit_mdl_fn(model_id="Baseline", 
#                             model_method=ifelse(glb_is_regression, "lm", 
#                                         ifelse(glb_is_binomial, "glm", "rpart")),
#                             indep_vars_vctr=glb_Baseline_mdl_var,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=0, tune_models_df=NULL,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
    ret_lst <- myfit_mdl_fn(model_id="Baseline", model_method="mybaseln_classfr",
                            indep_vars_vctr=glb_Baseline_mdl_var,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
}

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
ret_lst <- myfit_mdl(model_id="MFO", 
                     model_method=ifelse(glb_is_regression, "lm", "myMFO_classfr"), 
                     model_type=glb_model_type,
                        indep_vars_vctr=".rnorm",
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## Loading required package: caret
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```
## [1] "fitting model: MFO.myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] A B P R
## Levels: A B P R
## [1] "unique.prob:"
## y
##         P         A         B         R 
## 0.2580231 0.2528883 0.2458280 0.2432606 
## [1] "MFO.val:"
## [1] "P"
##             Length Class      Mode     
## unique.vals 4      factor     numeric  
## unique.prob 4      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   4      -none-     character
## [1] "in MFO.Classifier$predict"
##          Prediction
## Reference   A   B   P   R
##         A   0   0 394   0
##         B   0   0 383   0
##         P   0   0 402   0
##         R   0   0 379   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2580231      0.0000000      0.2364483      0.2805136      0.2580231 
## AccuracyPValue  McnemarPValue 
##      0.5096856            NaN 
## [1] "in MFO.Classifier$predict"
##          Prediction
## Reference   A   B   P   R
##         A   0   0 395   0
##         B   0   0 383   0
##         P   0   0 401   0
##         R   0   0 379   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2573813      0.0000000      0.2358253      0.2798554      0.2573813 
## AccuracyPValue  McnemarPValue 
##      0.5096885            NaN 
##            model_id  model_method  feats max.nTuningRuns
## 1 MFO.myMFO_classfr myMFO_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      0.398                 0.002        0.2580231
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.2364483             0.2805136             0
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.2573813             0.2358253             0.2798554
##   max.Kappa.OOB
## 1             0
```

```r
if (glb_is_classification)
    # "random" model - only for classification; none needed for regression since it is same as MFO
    ret_lst <- myfit_mdl(model_id="Random", model_method="myrandom_classfr",
                            model_type=glb_model_type,                         
                            indep_vars_vctr=".rnorm",
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Random.myrandom_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
##             Length Class      Mode     
## unique.vals 4      factor     numeric  
## unique.prob 4      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   4      -none-     character
##          Prediction
## Reference   A   B   P   R
##         A  93 108  93 100
##         B  95 105 104  79
##         P 118  99 102  83
##         R 108  99  89  83
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.245827985   -0.005794174    0.224621846    0.267996239    0.258023107 
## AccuracyPValue  McnemarPValue 
##    0.870854366    0.351265364 
##          Prediction
## Reference   A   B   P   R
##         A 103 106  99  87
##         B  94  87 106  96
##         P 103  94 100 104
##         R  88  96  98  97
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.248395379   -0.002275479    0.227109736    0.270633355    0.257381258 
## AccuracyPValue  McnemarPValue 
##    0.799231727    0.944878595 
##                  model_id     model_method  feats max.nTuningRuns
## 1 Random.myrandom_classfr myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      0.224                 0.001         0.245828
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.2246218             0.2679962  -0.005794174
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.2483954             0.2271097             0.2706334
##   max.Kappa.OOB
## 1  -0.002275479
```

```r
# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Max.cor.Y.cv.0.rpart"
## [1] "    indep_vars: ybar"
```

```
## Loading required package: rpart
```

```
## Fitting cp = 0.28 on full training set
```

```
## Loading required package: rpart.plot
```

![](Letter_Recognition_files/figure-html/fit.models_0-1.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##          CP nsplit rel error
## 1 0.2802768      0         1
## 
## Node number 1: 1558 observations
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 1156 P (0.2528883 0.2458280 0.2580231 0.2432606) *
##          Prediction
## Reference   A   B   P   R
##         A   0   0 394   0
##         B   0   0 383   0
##         P   0   0 402   0
##         R   0   0 379   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2580231      0.0000000      0.2364483      0.2805136      0.2580231 
## AccuracyPValue  McnemarPValue 
##      0.5096856            NaN 
##          Prediction
## Reference   A   B   P   R
##         A   0   0 395   0
##         B   0   0 383   0
##         P   0   0 401   0
##         R   0   0 379   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2573813      0.0000000      0.2358253      0.2798554      0.2573813 
## AccuracyPValue  McnemarPValue 
##      0.5096885            NaN 
##               model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.rpart        rpart  ybar               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      0.565                 0.025        0.2580231
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.2364483             0.2805136             0
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.2573813             0.2358253             0.2798554
##   max.Kappa.OOB
## 1             0
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0.cp.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
```

```
## [1] "fitting model: Max.cor.Y.cv.0.cp.0.rpart"
## [1] "    indep_vars: ybar"
## Fitting cp = 0 on full training set
```

![](Letter_Recognition_files/figure-html/fit.models_0-2.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##           CP nsplit rel error
## 1 0.28027682      0 1.0000000
## 2 0.25778547      1 0.7197232
## 3 0.01816609      2 0.4619377
## 4 0.00000000      3 0.4437716
## 
## Variable importance
## ybar 
##  100 
## 
## Node number 1: 1558 observations,    complexity param=0.2802768
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (351 obs) right son=3 (1207 obs)
##   Primary splits:
##       ybar < 5.5 to the left,  improve=287.8322, (0 missing)
## 
## Node number 2: 351 observations
##   predicted class=A  expected loss=0.05698006  P(node) =0.2252888
##     class counts:   331    10     7     3
##    probabilities: 0.943 0.028 0.020 0.009 
## 
## Node number 3: 1207 observations,    complexity param=0.2577855
##   predicted class=P  expected loss=0.6727423  P(node) =0.7747112
##     class counts:    63   373   395   376
##    probabilities: 0.052 0.309 0.327 0.312 
##   left son=6 (728 obs) right son=7 (479 obs)
##   Primary splits:
##       ybar < 8.5 to the left,  improve=174.7604, (0 missing)
## 
## Node number 6: 728 observations,    complexity param=0.01816609
##   predicted class=B  expected loss=0.5082418  P(node) =0.4672657
##     class counts:    60   358    60   250
##    probabilities: 0.082 0.492 0.082 0.343 
##   left son=12 (471 obs) right son=13 (257 obs)
##   Primary splits:
##       ybar < 7.5 to the left,  improve=12.57262, (0 missing)
## 
## Node number 7: 479 observations
##   predicted class=P  expected loss=0.3006263  P(node) =0.3074454
##     class counts:     3    15   335   126
##    probabilities: 0.006 0.031 0.699 0.263 
## 
## Node number 12: 471 observations
##   predicted class=B  expected loss=0.4288747  P(node) =0.3023107
##     class counts:    38   269    24   140
##    probabilities: 0.081 0.571 0.051 0.297 
## 
## Node number 13: 257 observations
##   predicted class=R  expected loss=0.5719844  P(node) =0.1649551
##     class counts:    22    89    36   110
##    probabilities: 0.086 0.346 0.140 0.428 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1558 1156 P (0.252888318 0.245827985 0.258023107 0.243260591)  
##    2) ybar< 5.5 351   20 A (0.943019943 0.028490028 0.019943020 0.008547009) *
##    3) ybar>=5.5 1207  812 P (0.052195526 0.309030655 0.327257664 0.311516156)  
##      6) ybar< 8.5 728  370 B (0.082417582 0.491758242 0.082417582 0.343406593)  
##       12) ybar< 7.5 471  202 B (0.080679406 0.571125265 0.050955414 0.297239915) *
##       13) ybar>=7.5 257  147 R (0.085603113 0.346303502 0.140077821 0.428015564) *
##      7) ybar>=8.5 479  144 P (0.006263048 0.031315240 0.699373695 0.263048017) *
##          Prediction
## Reference   A   B   P   R
##         A 331  38   3  22
##         B  10 269  15  89
##         P   7  24 335  36
##         R   3 140 126 110
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.707317e-01   5.605394e-01   6.467703e-01   6.940479e-01   2.580231e-01 
## AccuracyPValue  McnemarPValue 
##  7.252948e-255   1.876410e-18 
##          Prediction
## Reference   A   B   P   R
##         A 336  36   3  20
##         B   9 268  16  90
##         P   3  37 320  41
##         R   8 152 105 114
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.662388e-01   5.547216e-01   6.422061e-01   6.896432e-01   2.573813e-01 
## AccuracyPValue  McnemarPValue 
##  1.889326e-250   7.441567e-14 
##                    model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.cp.0.rpart        rpart  ybar               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                       0.46                 0.023        0.6707317
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.6467703             0.6940479     0.5605394
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.6662388             0.6422061             0.6896432
##   max.Kappa.OOB
## 1     0.5547216
```

```r
if (glb_is_regression || glb_is_binomial) # For multinomials this model will be run next by default
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)

# Used to compare vs. Interactions.High.cor.Y 
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.rpart"
## [1] "    indep_vars: ybar"
## + Fold1: cp=0.01817 
## - Fold1: cp=0.01817 
## + Fold2: cp=0.01817 
## - Fold2: cp=0.01817 
## + Fold3: cp=0.01817 
## - Fold3: cp=0.01817 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0182 on full training set
```

```
## Warning in myfit_mdl(model_id = "Max.cor.Y", model_method =
## ifelse(glb_is_regression, : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

![](Letter_Recognition_files/figure-html/fit.models_0-3.png) ![](Letter_Recognition_files/figure-html/fit.models_0-4.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##           CP nsplit rel error
## 1 0.28027682      0 1.0000000
## 2 0.25778547      1 0.7197232
## 3 0.01816609      2 0.4619377
## 
## Variable importance
## ybar 
##  100 
## 
## Node number 1: 1558 observations,    complexity param=0.2802768
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (351 obs) right son=3 (1207 obs)
##   Primary splits:
##       ybar < 5.5 to the left,  improve=287.8322, (0 missing)
## 
## Node number 2: 351 observations
##   predicted class=A  expected loss=0.05698006  P(node) =0.2252888
##     class counts:   331    10     7     3
##    probabilities: 0.943 0.028 0.020 0.009 
## 
## Node number 3: 1207 observations,    complexity param=0.2577855
##   predicted class=P  expected loss=0.6727423  P(node) =0.7747112
##     class counts:    63   373   395   376
##    probabilities: 0.052 0.309 0.327 0.312 
##   left son=6 (728 obs) right son=7 (479 obs)
##   Primary splits:
##       ybar < 8.5 to the left,  improve=174.7604, (0 missing)
## 
## Node number 6: 728 observations
##   predicted class=B  expected loss=0.5082418  P(node) =0.4672657
##     class counts:    60   358    60   250
##    probabilities: 0.082 0.492 0.082 0.343 
## 
## Node number 7: 479 observations
##   predicted class=P  expected loss=0.3006263  P(node) =0.3074454
##     class counts:     3    15   335   126
##    probabilities: 0.006 0.031 0.699 0.263 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 1156 P (0.252888318 0.245827985 0.258023107 0.243260591)  
##   2) ybar< 5.5 351   20 A (0.943019943 0.028490028 0.019943020 0.008547009) *
##   3) ybar>=5.5 1207  812 P (0.052195526 0.309030655 0.327257664 0.311516156)  
##     6) ybar< 8.5 728  370 B (0.082417582 0.491758242 0.082417582 0.343406593) *
##     7) ybar>=8.5 479  144 P (0.006263048 0.031315240 0.699373695 0.263048017) *
##          Prediction
## Reference   A   B   P   R
##         A 331  60   3   0
##         B  10 358  15   0
##         P   7  60 335   0
##         R   3 250 126   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.572529e-01   5.422911e-01   6.330850e-01   6.808266e-01   2.580231e-01 
## AccuracyPValue  McnemarPValue 
##  5.101688e-239   1.348274e-92 
##          Prediction
## Reference   A   B   P   R
##         A 336  56   3   0
##         B   9 358  16   0
##         P   3  78 320   0
##         R   8 266 105   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.508344e-01   5.338987e-01   6.265756e-01   6.745233e-01   2.573813e-01 
## AccuracyPValue  McnemarPValue 
##  1.475475e-232   7.181418e-95 
##          model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.rpart        rpart  ybar               3
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      1.041                 0.025        0.6598241
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1              0.633085             0.6808266     0.5457342
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.6508344             0.6265756             0.6745233
##   max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1     0.5338987         0.01444694      0.01906184
```

```r
# Interactions.High.cor.Y
if (nrow(int_feats_df <- subset(glb_feats_df, (cor.low == 0) & 
                                              (exclude.as.feat == 0))) > 0) {
    # lm & glm handle interaction terms; rpart & rf do not
    #   This does not work - why ???
#     indep_vars_vctr <- ifelse(glb_is_binomial, 
#         c(max_cor_y_x_var, paste(max_cor_y_x_var, 
#                         subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":")),
#         union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"]))
    if (glb_is_regression || glb_is_binomial) {
        indep_vars_vctr <- 
            c(max_cor_y_x_var, paste(max_cor_y_x_var, int_feats_df[, "id"], sep=":"))       
    } else { indep_vars_vctr <- union(max_cor_y_x_var, int_feats_df[, "id"]) }
    
    ret_lst <- myfit_mdl(model_id="Interact.High.cor.y", 
                            model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                         model_type=glb_model_type,
                            indep_vars_vctr,
                            glb_rsp_var, glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                            n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)                        
}    
```

```
## [1] "fitting model: Interact.High.cor.y.rpart"
## [1] "    indep_vars: ybar, xedgeycor, width, height, ybox"
## + Fold1: cp=0.1869 
## - Fold1: cp=0.1869 
## + Fold2: cp=0.1869 
## - Fold2: cp=0.1869 
## + Fold3: cp=0.1869 
## - Fold3: cp=0.1869 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.187 on full training set
```

```
## Warning in myfit_mdl(model_id = "Interact.High.cor.y", model_method =
## ifelse(glb_is_regression, : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

![](Letter_Recognition_files/figure-html/fit.models_0-5.png) ![](Letter_Recognition_files/figure-html/fit.models_0-6.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##          CP nsplit rel error
## 1 0.3192042      0 1.0000000
## 2 0.2586505      1 0.6807958
## 3 0.1868512      2 0.4221453
## 
## Variable importance
##      ybar xedgeycor    height 
##        51        48         1 
## 
## Node number 1: 1558 observations,    complexity param=0.3192042
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (1088 obs) right son=3 (470 obs)
##   Primary splits:
##       xedgeycor < 8.5  to the left,  improve=293.201000, (0 missing)
##       ybar      < 5.5  to the left,  improve=287.832200, (0 missing)
##       height    < 8.5  to the left,  improve= 11.995750, (0 missing)
##       width     < 7.5  to the left,  improve=  5.676199, (0 missing)
##       ybox      < 13.5 to the left,  improve=  1.984283, (0 missing)
##   Surrogate splits:
##       ybar   < 8.5  to the left,  agree=0.821, adj=0.406, (0 split)
##       height < 8.5  to the left,  agree=0.709, adj=0.034, (0 split)
##       width  < 8.5  to the left,  agree=0.700, adj=0.006, (0 split)
## 
## Node number 2: 1088 observations,    complexity param=0.2586505
##   predicted class=A  expected loss=0.6488971  P(node) =0.6983312
##     class counts:   382   338    13   355
##    probabilities: 0.351 0.311 0.012 0.326 
##   left son=4 (344 obs) right son=5 (744 obs)
##   Primary splits:
##       ybar      < 5.5  to the left,  improve=275.7625000, (0 missing)
##       xedgeycor < 7.5  to the right, improve=171.4917000, (0 missing)
##       width     < 1.5  to the right, improve=  2.4355750, (0 missing)
##       ybox      < 14.5 to the left,  improve=  0.7367482, (0 missing)
##       height    < 5.5  to the left,  improve=  0.4981618, (0 missing)
##   Surrogate splits:
##       xedgeycor < 6.5  to the left,  agree=0.773, adj=0.282, (0 split)
## 
## Node number 3: 470 observations
##   predicted class=P  expected loss=0.1723404  P(node) =0.3016688
##     class counts:    12    45   389    24
##    probabilities: 0.026 0.096 0.828 0.051 
## 
## Node number 4: 344 observations
##   predicted class=A  expected loss=0.04360465  P(node) =0.2207959
##     class counts:   329     9     3     3
##    probabilities: 0.956 0.026 0.009 0.009 
## 
## Node number 5: 744 observations
##   predicted class=R  expected loss=0.5268817  P(node) =0.4775353
##     class counts:    53   329    10   352
##    probabilities: 0.071 0.442 0.013 0.473 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 1156 P (0.25288832 0.24582798 0.25802311 0.24326059)  
##   2) xedgeycor< 8.5 1088  706 A (0.35110294 0.31066176 0.01194853 0.32628676)  
##     4) ybar< 5.5 344   15 A (0.95639535 0.02616279 0.00872093 0.00872093) *
##     5) ybar>=5.5 744  392 R (0.07123656 0.44220430 0.01344086 0.47311828) *
##   3) xedgeycor>=8.5 470   81 P (0.02553191 0.09574468 0.82765957 0.05106383) *
##          Prediction
## Reference   A   B   P   R
##         A 329   0  12  53
##         B   9   0  45 329
##         P   3   0 389  10
##         R   3   0  24 352
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.867779e-01   5.824598e-01   6.630906e-01   7.097592e-01   2.580231e-01 
## AccuracyPValue  McnemarPValue 
##  1.857237e-274   1.257666e-91 
##          Prediction
## Reference   A   B   P   R
##         A 335   0  17  43
##         B   8   0  47 328
##         P   2   0 387  12
##         R   8   0  26 345
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.848524e-01   5.798532e-01   6.611305e-01   7.078755e-01   2.573813e-01 
## AccuracyPValue  McnemarPValue 
##  4.921481e-273   1.910253e-88 
##                    model_id model_method
## 1 Interact.High.cor.y.rpart        rpart
##                                  feats max.nTuningRuns
## 1 ybar, xedgeycor, width, height, ybox               3
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      1.238                 0.046        0.7721086
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.6630906             0.7097592     0.6961876
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.6848524             0.6611305             0.7078755
##   max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1     0.5798532          0.0929909       0.1236135
```

```r
# Low.cor.X
ret_lst <- myfit_mdl(model_id="Low.cor.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                        indep_vars_vctr=subset(glb_feats_df, cor.low == 1)[, "id"],
                         model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Low.cor.X.rpart"
## [1] "    indep_vars: ybar, x2bar, x2ybar, y2bar, yedgexcor, yedge, xbox, onpix, xybar, xedge, xy2bar, xbar"
## + Fold1: cp=0.1315 
## - Fold1: cp=0.1315 
## + Fold2: cp=0.1315 
## - Fold2: cp=0.1315 
## + Fold3: cp=0.1315 
## - Fold3: cp=0.1315 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.131 on full training set
```

```
## Warning in myfit_mdl(model_id = "Low.cor.X", model_method =
## ifelse(glb_is_regression, : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

![](Letter_Recognition_files/figure-html/fit.models_0-7.png) ![](Letter_Recognition_files/figure-html/fit.models_0-8.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##          CP nsplit rel error
## 1 0.2802768      0 1.0000000
## 2 0.2768166      1 0.7197232
## 3 0.1314879      2 0.4429066
## 
## Variable importance
##   ybar xy2bar x2ybar  yedge   xbar  xedge  xybar  x2bar  y2bar 
##     24     16     12     10      9      9      7      7      5 
## 
## Node number 1: 1558 observations,    complexity param=0.2802768
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (351 obs) right son=3 (1207 obs)
##   Primary splits:
##       ybar   < 5.5  to the left,  improve=287.8322, (0 missing)
##       xy2bar < 5.5  to the right, improve=278.1742, (0 missing)
##       x2ybar < 2.5  to the left,  improve=262.6356, (0 missing)
##       yedge  < 4.5  to the left,  improve=177.0582, (0 missing)
##       y2bar  < 2.5  to the left,  improve=168.4795, (0 missing)
##   Surrogate splits:
##       x2ybar < 2.5  to the left,  agree=0.929, adj=0.684, (0 split)
##       x2bar  < 2.5  to the left,  agree=0.859, adj=0.376, (0 split)
##       y2bar  < 2.5  to the left,  agree=0.845, adj=0.311, (0 split)
##       yedge  < 2.5  to the left,  agree=0.841, adj=0.296, (0 split)
##       xbar   < 9.5  to the right, agree=0.839, adj=0.285, (0 split)
## 
## Node number 2: 351 observations
##   predicted class=A  expected loss=0.05698006  P(node) =0.2252888
##     class counts:   331    10     7     3
##    probabilities: 0.943 0.028 0.020 0.009 
## 
## Node number 3: 1207 observations,    complexity param=0.2768166
##   predicted class=P  expected loss=0.6727423  P(node) =0.7747112
##     class counts:    63   373   395   376
##    probabilities: 0.052 0.309 0.327 0.312 
##   left son=6 (835 obs) right son=7 (372 obs)
##   Primary splits:
##       xy2bar < 5.5  to the right, improve=263.6597, (0 missing)
##       ybar   < 8.5  to the left,  improve=174.7604, (0 missing)
##       xedge  < 1.5  to the right, improve=156.0710, (0 missing)
##       yedge  < 4.5  to the right, improve=139.6561, (0 missing)
##       xybar  < 10.5 to the left,  improve=131.9661, (0 missing)
##   Surrogate splits:
##       xedge < 1.5  to the right, agree=0.862, adj=0.554, (0 split)
##       xybar < 10.5 to the left,  agree=0.831, adj=0.452, (0 split)
##       ybar  < 8.5  to the left,  agree=0.815, adj=0.401, (0 split)
##       yedge < 3.5  to the right, agree=0.781, adj=0.290, (0 split)
##       xbar  < 5.5  to the right, agree=0.772, adj=0.261, (0 split)
## 
## Node number 6: 835 observations
##   predicted class=R  expected loss=0.5497006  P(node) =0.5359435
##     class counts:    62   341    56   376
##    probabilities: 0.074 0.408 0.067 0.450 
## 
## Node number 7: 372 observations
##   predicted class=P  expected loss=0.08870968  P(node) =0.2387677
##     class counts:     1    32   339     0
##    probabilities: 0.003 0.086 0.911 0.000 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 1156 P (0.252888318 0.245827985 0.258023107 0.243260591)  
##   2) ybar< 5.5 351   20 A (0.943019943 0.028490028 0.019943020 0.008547009) *
##   3) ybar>=5.5 1207  812 P (0.052195526 0.309030655 0.327257664 0.311516156)  
##     6) xy2bar>=5.5 835  459 R (0.074251497 0.408383234 0.067065868 0.450299401) *
##     7) xy2bar< 5.5 372   33 P (0.002688172 0.086021505 0.911290323 0.000000000) *
##          Prediction
## Reference   A   B   P   R
##         A 331   0   1  62
##         B  10   0  32 341
##         P   7   0 339  56
##         R   3   0   0 376
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.713736e-01   5.624414e-01   6.474225e-01   6.946769e-01   2.580231e-01 
## AccuracyPValue  McnemarPValue 
##  1.236260e-255  3.625271e-104 
##          Prediction
## Reference   A   B   P   R
##         A 336   0   2  57
##         B   9   0  22 352
##         P   3   0 330  68
##         R   8   0   0 371
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.655969e-01   5.548322e-01   6.415543e-01   6.890137e-01   2.573813e-01 
## AccuracyPValue  McnemarPValue 
##  1.086722e-249  3.017413e-102 
##          model_id model_method
## 1 Low.cor.X.rpart        rpart
##                                                                                   feats
## 1 ybar, x2bar, x2ybar, y2bar, yedgexcor, yedge, xbox, onpix, xybar, xedge, xy2bar, xbar
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      1.403                 0.081
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.7291809             0.6474225             0.6946769
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1     0.6390866        0.6655969             0.6415543
##   max.AccuracyUpper.OOB max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1             0.6890137     0.5548322         0.05790668       0.0770877
```

```r
# User specified
for (method in glb_models_method_vctr) {
    print(sprintf("iterating over method:%s", method))

    # All X that is not user excluded
    indep_vars_vctr <- setdiff(names(glb_trnent_df), 
        union(glb_rsp_var, glb_exclude_vars_as_features))
    
    # easier to exclude features
#     indep_vars_vctr <- setdiff(names(glb_trnent_df), 
#         union(union(glb_rsp_var, glb_exclude_vars_as_features), 
#               c("<feat1_name>", "<feat2_name>")))
    
    # easier to include features
#     indep_vars_vctr <- c("<feat1_name>", "<feat2_name>")

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glb_trnent_df), 
#                          union(glb_rsp_var, glb_exclude_vars_as_features)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]

#     glb_sel_mdl <- glb_sel_wlm_mdl <- ret_lst[["model"]]
#     rpart_sel_wlm_mdl <- rpart(reformulate(indep_vars_vctr, response=glb_rsp_var), 
#                                data=glb_trnent_df, method="class", 
#                                control=rpart.control(cp=glb_sel_wlm_mdl$bestTune$cp),
#                            parms=list(loss=glb_model_metric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 
    model_id_pfx <- "All.X";
    ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ""), model_method=method,
                            indep_vars_vctr=indep_vars_vctr,
                            model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
    
    # Since caret does not optimize rpart well
    if (method == "rpart")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".cp.0"), model_method=method,
                                indep_vars_vctr=indep_vars_vctr,
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,        
            n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
    
    # rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #   only for OOB in trainControl ?

#     ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
}
```

```
## [1] "iterating over method:rpart"
## [1] "fitting model: All.X.rpart"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor"
## + Fold1: cp=0.1869 
## - Fold1: cp=0.1869 
## + Fold2: cp=0.1869 
## - Fold2: cp=0.1869 
## + Fold3: cp=0.1869 
## - Fold3: cp=0.1869 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.187 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## cp
```

![](Letter_Recognition_files/figure-html/fit.models_0-9.png) ![](Letter_Recognition_files/figure-html/fit.models_0-10.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##          CP nsplit rel error
## 1 0.3192042      0 1.0000000
## 2 0.2586505      1 0.6807958
## 3 0.1868512      2 0.4221453
## 
## Variable importance
##      ybar xedgeycor    x2ybar    xy2bar     y2bar     yedge     x2bar 
##        21        15        14        10        10         9         6 
##     xedge     xybar      xbar 
##         6         4         4 
## 
## Node number 1: 1558 observations,    complexity param=0.3192042
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (1088 obs) right son=3 (470 obs)
##   Primary splits:
##       xedgeycor < 8.5  to the left,  improve=293.2010, (0 missing)
##       ybar      < 5.5  to the left,  improve=287.8322, (0 missing)
##       xy2bar    < 5.5  to the right, improve=278.1742, (0 missing)
##       x2ybar    < 2.5  to the left,  improve=262.6356, (0 missing)
##       yedge     < 4.5  to the left,  improve=177.0582, (0 missing)
##   Surrogate splits:
##       xy2bar < 5.5  to the right, agree=0.892, adj=0.643, (0 split)
##       ybar   < 8.5  to the left,  agree=0.821, adj=0.406, (0 split)
##       xedge  < 1.5  to the right, agree=0.816, adj=0.391, (0 split)
##       xybar  < 10.5 to the left,  agree=0.785, adj=0.287, (0 split)
##       x2ybar < 6.5  to the left,  agree=0.777, adj=0.262, (0 split)
## 
## Node number 2: 1088 observations,    complexity param=0.2586505
##   predicted class=A  expected loss=0.6488971  P(node) =0.6983312
##     class counts:   382   338    13   355
##    probabilities: 0.351 0.311 0.012 0.326 
##   left son=4 (344 obs) right son=5 (744 obs)
##   Primary splits:
##       ybar      < 5.5  to the left,  improve=275.7625, (0 missing)
##       x2ybar    < 2.5  to the left,  improve=240.6702, (0 missing)
##       y2bar     < 2.5  to the left,  improve=226.4519, (0 missing)
##       yedge     < 3.5  to the left,  improve=215.2610, (0 missing)
##       xedgeycor < 7.5  to the right, improve=171.4917, (0 missing)
##   Surrogate splits:
##       x2ybar < 2.5  to the left,  agree=0.904, adj=0.698, (0 split)
##       y2bar  < 2.5  to the left,  agree=0.892, adj=0.657, (0 split)
##       yedge  < 3.5  to the left,  agree=0.881, adj=0.625, (0 split)
##       x2bar  < 2.5  to the left,  agree=0.820, adj=0.430, (0 split)
##       xbar   < 9.5  to the right, agree=0.779, adj=0.302, (0 split)
## 
## Node number 3: 470 observations
##   predicted class=P  expected loss=0.1723404  P(node) =0.3016688
##     class counts:    12    45   389    24
##    probabilities: 0.026 0.096 0.828 0.051 
## 
## Node number 4: 344 observations
##   predicted class=A  expected loss=0.04360465  P(node) =0.2207959
##     class counts:   329     9     3     3
##    probabilities: 0.956 0.026 0.009 0.009 
## 
## Node number 5: 744 observations
##   predicted class=R  expected loss=0.5268817  P(node) =0.4775353
##     class counts:    53   329    10   352
##    probabilities: 0.071 0.442 0.013 0.473 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 1156 P (0.25288832 0.24582798 0.25802311 0.24326059)  
##   2) xedgeycor< 8.5 1088  706 A (0.35110294 0.31066176 0.01194853 0.32628676)  
##     4) ybar< 5.5 344   15 A (0.95639535 0.02616279 0.00872093 0.00872093) *
##     5) ybar>=5.5 744  392 R (0.07123656 0.44220430 0.01344086 0.47311828) *
##   3) xedgeycor>=8.5 470   81 P (0.02553191 0.09574468 0.82765957 0.05106383) *
##          Prediction
## Reference   A   B   P   R
##         A 329   0  12  53
##         B   9   0  45 329
##         P   3   0 389  10
##         R   3   0  24 352
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.867779e-01   5.824598e-01   6.630906e-01   7.097592e-01   2.580231e-01 
## AccuracyPValue  McnemarPValue 
##  1.857237e-274   1.257666e-91 
##          Prediction
## Reference   A   B   P   R
##         A 335   0  17  43
##         B   8   0  47 328
##         P   2   0 387  12
##         R   8   0  26 345
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.848524e-01   5.798532e-01   6.611305e-01   7.078755e-01   2.573813e-01 
## AccuracyPValue  McnemarPValue 
##  4.921481e-273   1.910253e-88 
##      model_id model_method
## 1 All.X.rpart        rpart
##                                                                                                                   feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      1.239                 0.096
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.7721086             0.6630906             0.7097592
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1     0.6961876        0.6848524             0.6611305
##   max.AccuracyUpper.OOB max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1             0.7078755     0.5798532          0.0929909       0.1236135
## [1] "fitting model: All.X.cp.0.rpart"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor"
## Fitting cp = 0 on full training set
```

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##              CP nsplit  rel error
## 1  0.3192041522      0 1.00000000
## 2  0.2586505190      1 0.68079585
## 3  0.1868512111      2 0.42214533
## 4  0.0259515571      3 0.23529412
## 5  0.0207612457      4 0.20934256
## 6  0.0173010381      5 0.18858131
## 7  0.0138408304      6 0.17128028
## 8  0.0121107266      7 0.15743945
## 9  0.0077854671      8 0.14532872
## 10 0.0060553633     11 0.12197232
## 11 0.0051903114     14 0.10380623
## 12 0.0043252595     15 0.09861592
## 13 0.0025951557     16 0.09429066
## 14 0.0017301038     17 0.09169550
## 15 0.0015138408     18 0.08996540
## 16 0.0008650519     22 0.08391003
## 17 0.0002883506     23 0.08304498
## 18 0.0000000000     26 0.08217993
## 
## Variable importance
##      ybar xedgeycor    x2ybar    xy2bar     yedge     y2bar     xedge 
##        16        15        14        11        11         8         8 
##     xybar     x2bar      xbar yedgexcor      ybox      xbox 
##         6         5         4         1         1         1 
## 
## Node number 1: 1558 observations,    complexity param=0.3192042
##   predicted class=P  expected loss=0.7419769  P(node) =1
##     class counts:   394   383   402   379
##    probabilities: 0.253 0.246 0.258 0.243 
##   left son=2 (1088 obs) right son=3 (470 obs)
##   Primary splits:
##       xedgeycor < 8.5  to the left,  improve=293.2010, (0 missing)
##       ybar      < 5.5  to the left,  improve=287.8322, (0 missing)
##       xy2bar    < 5.5  to the right, improve=278.1742, (0 missing)
##       x2ybar    < 2.5  to the left,  improve=262.6356, (0 missing)
##       yedge     < 4.5  to the left,  improve=177.0582, (0 missing)
##   Surrogate splits:
##       xy2bar < 5.5  to the right, agree=0.892, adj=0.643, (0 split)
##       ybar   < 8.5  to the left,  agree=0.821, adj=0.406, (0 split)
##       xedge  < 1.5  to the right, agree=0.816, adj=0.391, (0 split)
##       xybar  < 10.5 to the left,  agree=0.785, adj=0.287, (0 split)
##       x2ybar < 6.5  to the left,  agree=0.777, adj=0.262, (0 split)
## 
## Node number 2: 1088 observations,    complexity param=0.2586505
##   predicted class=A  expected loss=0.6488971  P(node) =0.6983312
##     class counts:   382   338    13   355
##    probabilities: 0.351 0.311 0.012 0.326 
##   left son=4 (344 obs) right son=5 (744 obs)
##   Primary splits:
##       ybar      < 5.5  to the left,  improve=275.7625, (0 missing)
##       x2ybar    < 2.5  to the left,  improve=240.6702, (0 missing)
##       y2bar     < 2.5  to the left,  improve=226.4519, (0 missing)
##       yedge     < 3.5  to the left,  improve=215.2610, (0 missing)
##       xedgeycor < 7.5  to the right, improve=171.4917, (0 missing)
##   Surrogate splits:
##       x2ybar < 2.5  to the left,  agree=0.904, adj=0.698, (0 split)
##       y2bar  < 2.5  to the left,  agree=0.892, adj=0.657, (0 split)
##       yedge  < 3.5  to the left,  agree=0.881, adj=0.625, (0 split)
##       x2bar  < 2.5  to the left,  agree=0.820, adj=0.430, (0 split)
##       xbar   < 9.5  to the right, agree=0.779, adj=0.302, (0 split)
## 
## Node number 3: 470 observations,    complexity param=0.01730104
##   predicted class=P  expected loss=0.1723404  P(node) =0.3016688
##     class counts:    12    45   389    24
##    probabilities: 0.026 0.096 0.828 0.051 
##   left son=6 (91 obs) right son=7 (379 obs)
##   Primary splits:
##       xybar  < 7.5  to the left,  improve=59.48719, (0 missing)
##       xy2bar < 6.5  to the right, improve=54.86112, (0 missing)
##       ybar   < 7.5  to the left,  improve=49.49367, (0 missing)
##       yedge  < 6.5  to the right, improve=48.42295, (0 missing)
##       xedge  < 5.5  to the left,  improve=30.83057, (0 missing)
##   Surrogate splits:
##       xy2bar < 6.5  to the right, agree=0.936, adj=0.670, (0 split)
##       ybar   < 7.5  to the left,  agree=0.902, adj=0.495, (0 split)
##       xedge  < 5.5  to the right, agree=0.889, adj=0.429, (0 split)
##       yedge  < 6.5  to the right, agree=0.885, adj=0.407, (0 split)
##       onpix  < 6.5  to the right, agree=0.838, adj=0.165, (0 split)
## 
## Node number 4: 344 observations,    complexity param=0.006055363
##   predicted class=A  expected loss=0.04360465  P(node) =0.2207959
##     class counts:   329     9     3     3
##    probabilities: 0.956 0.026 0.009 0.009 
##   left son=8 (328 obs) right son=9 (16 obs)
##   Primary splits:
##       y2bar     < 4.5  to the left,  improve=17.189240, (0 missing)
##       yedge     < 5.5  to the left,  improve=13.564310, (0 missing)
##       x2bar     < 4.5  to the left,  improve= 8.523218, (0 missing)
##       yedgexcor < 9.5  to the left,  improve= 7.120189, (0 missing)
##       xbox      < 7.5  to the left,  improve= 5.046520, (0 missing)
##   Surrogate splits:
##       yedge     < 5.5  to the left,  agree=0.971, adj=0.375, (0 split)
##       yedgexcor < 10.5 to the left,  agree=0.962, adj=0.188, (0 split)
##       xy2bar    < 5.5  to the right, agree=0.959, adj=0.125, (0 split)
##       xbox      < 7.5  to the left,  agree=0.956, adj=0.063, (0 split)
##       onpix     < 6.5  to the left,  agree=0.956, adj=0.063, (0 split)
## 
## Node number 5: 744 observations,    complexity param=0.1868512
##   predicted class=R  expected loss=0.5268817  P(node) =0.4775353
##     class counts:    53   329    10   352
##    probabilities: 0.071 0.442 0.013 0.473 
##   left son=10 (342 obs) right son=11 (402 obs)
##   Primary splits:
##       xedgeycor < 7.5  to the right, improve=139.70670, (0 missing)
##       xy2bar    < 7.5  to the left,  improve= 92.43059, (0 missing)
##       x2ybar    < 5.5  to the right, improve= 81.07422, (0 missing)
##       y2bar     < 4.5  to the right, improve= 56.45671, (0 missing)
##       yedgexcor < 10.5 to the left,  improve= 52.58754, (0 missing)
##   Surrogate splits:
##       x2ybar < 5.5  to the right, agree=0.738, adj=0.430, (0 split)
##       xy2bar < 6.5  to the left,  agree=0.675, adj=0.292, (0 split)
##       xedge  < 2.5  to the left,  agree=0.675, adj=0.292, (0 split)
##       yedge  < 5.5  to the right, agree=0.644, adj=0.225, (0 split)
##       ybar   < 7.5  to the left,  agree=0.625, adj=0.184, (0 split)
## 
## Node number 6: 91 observations,    complexity param=0.01384083
##   predicted class=B  expected loss=0.5604396  P(node) =0.05840822
##     class counts:    10    40    20    21
##    probabilities: 0.110 0.440 0.220 0.231 
##   left son=12 (55 obs) right son=13 (36 obs)
##   Primary splits:
##       x2bar     < 3.5  to the right, improve=14.308240, (0 missing)
##       xy2bar    < 7.5  to the left,  improve= 9.472092, (0 missing)
##       yedge     < 4.5  to the left,  improve= 9.449763, (0 missing)
##       x2ybar    < 7.5  to the right, improve= 8.053076, (0 missing)
##       yedgexcor < 6.5  to the right, improve= 7.478284, (0 missing)
##   Surrogate splits:
##       yedgexcor < 5.5  to the right, agree=0.736, adj=0.333, (0 split)
##       x2ybar    < 7.5  to the left,  agree=0.725, adj=0.306, (0 split)
##       yedge     < 5.5  to the right, agree=0.725, adj=0.306, (0 split)
##       xy2bar    < 8.5  to the left,  agree=0.714, adj=0.278, (0 split)
##       ybar      < 7.5  to the left,  agree=0.681, adj=0.194, (0 split)
## 
## Node number 7: 379 observations
##   predicted class=P  expected loss=0.02638522  P(node) =0.2432606
##     class counts:     2     5   369     3
##    probabilities: 0.005 0.013 0.974 0.008 
## 
## Node number 8: 328 observations
##   predicted class=A  expected loss=0.00304878  P(node) =0.2105263
##     class counts:   327     0     1     0
##    probabilities: 0.997 0.000 0.003 0.000 
## 
## Node number 9: 16 observations
##   predicted class=B  expected loss=0.4375  P(node) =0.01026958
##     class counts:     2     9     2     3
##    probabilities: 0.125 0.562 0.125 0.188 
## 
## Node number 10: 342 observations,    complexity param=0.02595156
##   predicted class=B  expected loss=0.2192982  P(node) =0.2195122
##     class counts:    14   267    10    51
##    probabilities: 0.041 0.781 0.029 0.149 
##   left son=20 (283 obs) right son=21 (59 obs)
##   Primary splits:
##       xy2bar    < 7.5  to the left,  improve=48.65030, (0 missing)
##       xedge     < 2.5  to the left,  improve=33.98799, (0 missing)
##       y2bar     < 4.5  to the right, improve=27.13499, (0 missing)
##       yedgexcor < 6.5  to the left,  improve=15.49245, (0 missing)
##       ybar      < 8.5  to the left,  improve=15.03303, (0 missing)
##   Surrogate splits:
##       xedge     < 5.5  to the left,  agree=0.871, adj=0.254, (0 split)
##       yedgexcor < 4.5  to the right, agree=0.854, adj=0.153, (0 split)
##       ybar      < 9.5  to the left,  agree=0.848, adj=0.119, (0 split)
##       xbox      < 6.5  to the left,  agree=0.842, adj=0.085, (0 split)
##       ybox      < 11.5 to the left,  agree=0.842, adj=0.085, (0 split)
## 
## Node number 11: 402 observations,    complexity param=0.02076125
##   predicted class=R  expected loss=0.2512438  P(node) =0.2580231
##     class counts:    39    62     0   301
##    probabilities: 0.097 0.154 0.000 0.749 
##   left son=22 (26 obs) right son=23 (376 obs)
##   Primary splits:
##       yedge     < 2.5  to the left,  improve=35.46191, (0 missing)
##       x2ybar    < 0.5  to the left,  improve=34.14932, (0 missing)
##       y2bar     < 1.5  to the left,  improve=33.87850, (0 missing)
##       x2bar     < 3.5  to the left,  improve=19.57685, (0 missing)
##       yedgexcor < 8.5  to the left,  improve=19.07812, (0 missing)
##   Surrogate splits:
##       y2bar  < 1.5  to the left,  agree=0.993, adj=0.885, (0 split)
##       x2ybar < 0.5  to the left,  agree=0.993, adj=0.885, (0 split)
## 
## Node number 12: 55 observations,    complexity param=0.00432526
##   predicted class=B  expected loss=0.3090909  P(node) =0.03530167
##     class counts:     1    38    13     3
##    probabilities: 0.018 0.691 0.236 0.055 
##   left son=24 (46 obs) right son=25 (9 obs)
##   Primary splits:
##       xedgeycor < 10.5 to the left,  improve=5.553711, (0 missing)
##       yedge     < 6.5  to the right, improve=3.923601, (0 missing)
##       y2bar     < 3.5  to the right, improve=3.388742, (0 missing)
##       ybox      < 5.5  to the right, improve=3.175433, (0 missing)
##       height    < 6.5  to the left,  improve=3.096970, (0 missing)
##   Surrogate splits:
##       yedge  < 5.5  to the right, agree=0.891, adj=0.333, (0 split)
##       height < 8.5  to the left,  agree=0.873, adj=0.222, (0 split)
##       ybox   < 5.5  to the right, agree=0.855, adj=0.111, (0 split)
##       x2ybar < 7.5  to the left,  agree=0.855, adj=0.111, (0 split)
## 
## Node number 13: 36 observations,    complexity param=0.007785467
##   predicted class=R  expected loss=0.5  P(node) =0.02310655
##     class counts:     9     2     7    18
##    probabilities: 0.250 0.056 0.194 0.500 
##   left son=26 (16 obs) right son=27 (20 obs)
##   Primary splits:
##       x2ybar < 6.5  to the right, improve=11.802780, (0 missing)
##       yedge  < 4.5  to the left,  improve= 9.862393, (0 missing)
##       xy2bar < 7.5  to the left,  improve= 7.379596, (0 missing)
##       onpix  < 4.5  to the left,  improve= 5.920635, (0 missing)
##       xbar   < 7.5  to the right, improve= 5.358045, (0 missing)
##   Surrogate splits:
##       yedge  < 5.5  to the left,  agree=0.833, adj=0.625, (0 split)
##       ybox   < 5.5  to the left,  agree=0.806, adj=0.562, (0 split)
##       ybar   < 7.5  to the right, agree=0.806, adj=0.562, (0 split)
##       xbar   < 7.5  to the right, agree=0.750, adj=0.437, (0 split)
##       xy2bar < 7.5  to the left,  agree=0.750, adj=0.437, (0 split)
## 
## Node number 20: 283 observations,    complexity param=0.0008650519
##   predicted class=B  expected loss=0.08480565  P(node) =0.1816431
##     class counts:     3   259     8    13
##    probabilities: 0.011 0.915 0.028 0.046 
##   left son=40 (265 obs) right son=41 (18 obs)
##   Primary splits:
##       xedge     < 4.5  to the left,  improve=9.331344, (0 missing)
##       xybar     < 10.5 to the left,  improve=4.976699, (0 missing)
##       y2bar     < 4.5  to the right, improve=4.621234, (0 missing)
##       yedgexcor < 8.5  to the right, improve=3.164135, (0 missing)
##       ybox      < 7.5  to the left,  improve=2.013517, (0 missing)
##   Surrogate splits:
##       onpix < 9.5  to the left,  agree=0.947, adj=0.167, (0 split)
##       y2bar < 3.5  to the right, agree=0.947, adj=0.167, (0 split)
## 
## Node number 21: 59 observations,    complexity param=0.006055363
##   predicted class=R  expected loss=0.3559322  P(node) =0.03786906
##     class counts:    11     8     2    38
##    probabilities: 0.186 0.136 0.034 0.644 
##   left son=42 (12 obs) right son=43 (47 obs)
##   Primary splits:
##       yedgexcor < 5.5  to the left,  improve=8.275935, (0 missing)
##       x2ybar    < 5.5  to the right, improve=7.481558, (0 missing)
##       xbar      < 7.5  to the right, improve=5.164139, (0 missing)
##       yedge     < 6.5  to the right, improve=5.098197, (0 missing)
##       ybar      < 8.5  to the left,  improve=5.072034, (0 missing)
##   Surrogate splits:
##       yedge < 6.5  to the right, agree=0.831, adj=0.167, (0 split)
## 
## Node number 22: 26 observations
##   predicted class=A  expected loss=0.03846154  P(node) =0.01668806
##     class counts:    25     0     0     1
##    probabilities: 0.962 0.000 0.000 0.038 
## 
## Node number 23: 376 observations,    complexity param=0.01211073
##   predicted class=R  expected loss=0.2021277  P(node) =0.241335
##     class counts:    14    62     0   300
##    probabilities: 0.037 0.165 0.000 0.798 
##   left son=46 (26 obs) right son=47 (350 obs)
##   Primary splits:
##       yedge  < 7.5  to the right, improve=19.73450, (0 missing)
##       x2ybar < 5.5  to the right, improve=16.32647, (0 missing)
##       xybar  < 8.5  to the right, improve=15.20779, (0 missing)
##       xedge  < 3.5  to the right, improve=14.35240, (0 missing)
##       onpix  < 4.5  to the right, improve=12.94437, (0 missing)
##   Surrogate splits:
##       xedgeycor < 4.5  to the left,  agree=0.939, adj=0.115, (0 split)
## 
## Node number 24: 46 observations
##   predicted class=B  expected loss=0.2173913  P(node) =0.02952503
##     class counts:     1    36     6     3
##    probabilities: 0.022 0.783 0.130 0.065 
## 
## Node number 25: 9 observations
##   predicted class=P  expected loss=0.2222222  P(node) =0.005776637
##     class counts:     0     2     7     0
##    probabilities: 0.000 0.222 0.778 0.000 
## 
## Node number 26: 16 observations
##   predicted class=A  expected loss=0.4375  P(node) =0.01026958
##     class counts:     9     0     7     0
##    probabilities: 0.562 0.000 0.438 0.000 
## 
## Node number 27: 20 observations
##   predicted class=R  expected loss=0.1  P(node) =0.01283697
##     class counts:     0     2     0    18
##    probabilities: 0.000 0.100 0.000 0.900 
## 
## Node number 40: 265 observations,    complexity param=0.0002883506
##   predicted class=B  expected loss=0.04528302  P(node) =0.1700899
##     class counts:     1   253     5     6
##    probabilities: 0.004 0.955 0.019 0.023 
##   left son=80 (254 obs) right son=81 (11 obs)
##   Primary splits:
##       xybar  < 10.5 to the left,  improve=4.1066790, (0 missing)
##       xedge  < 2.5  to the left,  improve=3.1094340, (0 missing)
##       y2bar  < 4.5  to the right, improve=1.0296950, (0 missing)
##       yedge  < 5.5  to the right, improve=0.9774409, (0 missing)
##       x2ybar < 4.5  to the right, improve=0.8363981, (0 missing)
##   Surrogate splits:
##       xy2bar < 4.5  to the right, agree=0.966, adj=0.182, (0 split)
## 
## Node number 41: 18 observations
##   predicted class=R  expected loss=0.6111111  P(node) =0.01155327
##     class counts:     2     6     3     7
##    probabilities: 0.111 0.333 0.167 0.389 
## 
## Node number 42: 12 observations
##   predicted class=A  expected loss=0.25  P(node) =0.007702182
##     class counts:     9     1     0     2
##    probabilities: 0.750 0.083 0.000 0.167 
## 
## Node number 43: 47 observations,    complexity param=0.002595156
##   predicted class=R  expected loss=0.2340426  P(node) =0.03016688
##     class counts:     2     7     2    36
##    probabilities: 0.043 0.149 0.043 0.766 
##   left son=86 (8 obs) right son=87 (39 obs)
##   Primary splits:
##       xbar   < 8.5  to the right, improve=4.834561, (0 missing)
##       x2ybar < 5.5  to the right, improve=3.043935, (0 missing)
##       ybar   < 8.5  to the left,  improve=2.569909, (0 missing)
##       x2bar  < 6.5  to the left,  improve=2.569909, (0 missing)
##       xybar  < 8.5  to the right, improve=1.905074, (0 missing)
##   Surrogate splits:
##       ybar  < 6.5  to the left,  agree=0.872, adj=0.250, (0 split)
##       xybar < 8.5  to the right, agree=0.872, adj=0.250, (0 split)
##       ybox  < 11.5 to the right, agree=0.851, adj=0.125, (0 split)
## 
## Node number 46: 26 observations,    complexity param=0.001730104
##   predicted class=B  expected loss=0.3076923  P(node) =0.01668806
##     class counts:     4    18     0     4
##    probabilities: 0.154 0.692 0.000 0.154 
##   left son=92 (16 obs) right son=93 (10 obs)
##   Primary splits:
##       xybar < 8.5  to the right, improve=5.907692, (0 missing)
##       x2bar < 5.5  to the left,  improve=4.647562, (0 missing)
##       xedge < 5.5  to the right, improve=4.647562, (0 missing)
##       xbox  < 6.5  to the right, improve=3.164835, (0 missing)
##       ybox  < 9.5  to the right, improve=3.000503, (0 missing)
##   Surrogate splits:
##       x2bar < 5.5  to the left,  agree=0.885, adj=0.7, (0 split)
##       xedge < 5.5  to the right, agree=0.885, adj=0.7, (0 split)
##       xbox  < 6.5  to the right, agree=0.846, adj=0.6, (0 split)
##       ybox  < 9.5  to the right, agree=0.808, adj=0.5, (0 split)
##       y2bar < 3.5  to the right, agree=0.769, adj=0.4, (0 split)
## 
## Node number 47: 350 observations,    complexity param=0.007785467
##   predicted class=R  expected loss=0.1542857  P(node) =0.224647
##     class counts:    10    44     0   296
##    probabilities: 0.029 0.126 0.000 0.846 
##   left son=94 (68 obs) right son=95 (282 obs)
##   Primary splits:
##       xybar  < 9.5  to the right, improve=13.955520, (0 missing)
##       x2ybar < 6.5  to the right, improve=10.014690, (0 missing)
##       x2bar  < 2.5  to the left,  improve= 9.697126, (0 missing)
##       xedge  < 3.5  to the right, improve= 7.913811, (0 missing)
##       ybar   < 6.5  to the left,  improve= 6.859111, (0 missing)
##   Surrogate splits:
##       xbar   < 9.5  to the right, agree=0.866, adj=0.309, (0 split)
##       x2bar  < 3.5  to the left,  agree=0.837, adj=0.162, (0 split)
##       x2ybar < 2.5  to the left,  agree=0.834, adj=0.147, (0 split)
##       y2bar  < 6.5  to the right, agree=0.820, adj=0.074, (0 split)
##       xbox   < 10.5 to the right, agree=0.817, adj=0.059, (0 split)
## 
## Node number 80: 254 observations,    complexity param=0.0002883506
##   predicted class=B  expected loss=0.02755906  P(node) =0.1630295
##     class counts:     1   247     0     6
##    probabilities: 0.004 0.972 0.000 0.024 
##   left son=160 (206 obs) right son=161 (48 obs)
##   Primary splits:
##       xedge     < 2.5  to the left,  improve=1.4530840, (0 missing)
##       yedgexcor < 10.5 to the left,  improve=0.9702039, (0 missing)
##       y2bar     < 4.5  to the right, improve=0.8697742, (0 missing)
##       xbar      < 6.5  to the right, improve=0.5396442, (0 missing)
##       ybox      < 7.5  to the left,  improve=0.3804576, (0 missing)
##   Surrogate splits:
##       ybox      < 10.5 to the left,  agree=0.827, adj=0.083, (0 split)
##       width     < 7.5  to the left,  agree=0.827, adj=0.083, (0 split)
##       yedgexcor < 8.5  to the right, agree=0.827, adj=0.083, (0 split)
##       onpix     < 7.5  to the left,  agree=0.823, adj=0.062, (0 split)
##       xbox      < 6.5  to the left,  agree=0.819, adj=0.042, (0 split)
## 
## Node number 81: 11 observations
##   predicted class=B  expected loss=0.4545455  P(node) =0.007060334
##     class counts:     0     6     5     0
##    probabilities: 0.000 0.545 0.455 0.000 
## 
## Node number 86: 8 observations
##   predicted class=B  expected loss=0.375  P(node) =0.005134788
##     class counts:     1     5     0     2
##    probabilities: 0.125 0.625 0.000 0.250 
## 
## Node number 87: 39 observations
##   predicted class=R  expected loss=0.1282051  P(node) =0.02503209
##     class counts:     1     2     2    34
##    probabilities: 0.026 0.051 0.051 0.872 
## 
## Node number 92: 16 observations
##   predicted class=B  expected loss=0  P(node) =0.01026958
##     class counts:     0    16     0     0
##    probabilities: 0.000 1.000 0.000 0.000 
## 
## Node number 93: 10 observations
##   predicted class=A  expected loss=0.6  P(node) =0.006418485
##     class counts:     4     2     0     4
##    probabilities: 0.400 0.200 0.000 0.400 
## 
## Node number 94: 68 observations,    complexity param=0.007785467
##   predicted class=R  expected loss=0.4264706  P(node) =0.0436457
##     class counts:     0    29     0    39
##    probabilities: 0.000 0.426 0.000 0.574 
##   left son=188 (40 obs) right son=189 (28 obs)
##   Primary splits:
##       yedge     < 4.5  to the right, improve=17.314710, (0 missing)
##       x2ybar    < 3.5  to the right, improve=14.057780, (0 missing)
##       yedgexcor < 9.5  to the left,  improve= 7.798303, (0 missing)
##       ybar      < 6.5  to the left,  improve= 5.370366, (0 missing)
##       xbar      < 9.5  to the left,  improve= 2.580911, (0 missing)
##   Surrogate splits:
##       xedge  < 3.5  to the right, agree=0.794, adj=0.500, (0 split)
##       xbox   < 4.5  to the right, agree=0.706, adj=0.286, (0 split)
##       ybox   < 8.5  to the right, agree=0.662, adj=0.179, (0 split)
##       x2ybar < 3.5  to the right, agree=0.662, adj=0.179, (0 split)
##       xbar   < 9.5  to the left,  agree=0.647, adj=0.143, (0 split)
## 
## Node number 95: 282 observations,    complexity param=0.005190311
##   predicted class=R  expected loss=0.08865248  P(node) =0.1810013
##     class counts:    10    15     0   257
##    probabilities: 0.035 0.053 0.000 0.911 
##   left son=190 (7 obs) right son=191 (275 obs)
##   Primary splits:
##       x2ybar    < 6.5  to the right, improve=10.866010, (0 missing)
##       yedgexcor < 6.5  to the left,  improve= 6.393287, (0 missing)
##       onpix     < 4.5  to the right, improve= 4.390620, (0 missing)
##       xedge     < 4.5  to the right, improve= 4.115344, (0 missing)
##       x2bar     < 3.5  to the left,  improve= 2.955054, (0 missing)
##   Surrogate splits:
##       x2bar < 2.5  to the left,  agree=0.993, adj=0.714, (0 split)
## 
## Node number 160: 206 observations
##   predicted class=B  expected loss=0  P(node) =0.1322208
##     class counts:     0   206     0     0
##    probabilities: 0.000 1.000 0.000 0.000 
## 
## Node number 161: 48 observations,    complexity param=0.0002883506
##   predicted class=B  expected loss=0.1458333  P(node) =0.03080873
##     class counts:     1    41     0     6
##    probabilities: 0.021 0.854 0.000 0.125 
##   left son=322 (39 obs) right son=323 (9 obs)
##   Primary splits:
##       xbar   < 6.5  to the right, improve=3.917735, (0 missing)
##       width  < 5.5  to the right, improve=2.402273, (0 missing)
##       xybar  < 6.5  to the right, improve=2.122863, (0 missing)
##       xy2bar < 6.5  to the left,  improve=1.719444, (0 missing)
##       x2bar  < 5.5  to the left,  improve=1.710979, (0 missing)
##   Surrogate splits:
##       x2bar < 7.5  to the left,  agree=0.896, adj=0.444, (0 split)
##       yedge < 8.5  to the left,  agree=0.854, adj=0.222, (0 split)
##       xybar < 6.5  to the right, agree=0.833, adj=0.111, (0 split)
## 
## Node number 188: 40 observations,    complexity param=0.006055363
##   predicted class=B  expected loss=0.275  P(node) =0.02567394
##     class counts:     0    29     0    11
##    probabilities: 0.000 0.725 0.000 0.275 
##   left son=376 (25 obs) right son=377 (15 obs)
##   Primary splits:
##       x2ybar    < 3.5  to the right, improve=10.083330, (0 missing)
##       xedgeycor < 6.5  to the right, improve= 4.571554, (0 missing)
##       xedge     < 4.5  to the left,  improve= 4.050000, (0 missing)
##       yedgexcor < 9.5  to the left,  improve= 3.152020, (0 missing)
##       ybox      < 10.5 to the left,  improve= 2.700000, (0 missing)
##   Surrogate splits:
##       xbar      < 9.5  to the left,  agree=0.800, adj=0.467, (0 split)
##       yedgexcor < 9.5  to the left,  agree=0.775, adj=0.400, (0 split)
##       width     < 7.5  to the left,  agree=0.750, adj=0.333, (0 split)
##       xedge     < 6.5  to the left,  agree=0.750, adj=0.333, (0 split)
##       xedgeycor < 6.5  to the right, agree=0.750, adj=0.333, (0 split)
## 
## Node number 189: 28 observations
##   predicted class=R  expected loss=0  P(node) =0.01797176
##     class counts:     0     0     0    28
##    probabilities: 0.000 0.000 0.000 1.000 
## 
## Node number 190: 7 observations
##   predicted class=A  expected loss=0.1428571  P(node) =0.00449294
##     class counts:     6     1     0     0
##    probabilities: 0.857 0.143 0.000 0.000 
## 
## Node number 191: 275 observations,    complexity param=0.001513841
##   predicted class=R  expected loss=0.06545455  P(node) =0.1765083
##     class counts:     4    14     0   257
##    probabilities: 0.015 0.051 0.000 0.935 
##   left son=382 (11 obs) right son=383 (264 obs)
##   Primary splits:
##       yedgexcor < 6.5  to the left,  improve=4.232727, (0 missing)
##       yedge     < 6.5  to the right, improve=3.519041, (0 missing)
##       x2ybar    < 5.5  to the right, improve=2.527475, (0 missing)
##       onpix     < 4.5  to the right, improve=2.260802, (0 missing)
##       ybox      < 7.5  to the right, improve=2.173986, (0 missing)
## 
## Node number 322: 39 observations
##   predicted class=B  expected loss=0.05128205  P(node) =0.02503209
##     class counts:     1    37     0     1
##    probabilities: 0.026 0.949 0.000 0.026 
## 
## Node number 323: 9 observations
##   predicted class=R  expected loss=0.4444444  P(node) =0.005776637
##     class counts:     0     4     0     5
##    probabilities: 0.000 0.444 0.000 0.556 
## 
## Node number 376: 25 observations
##   predicted class=B  expected loss=0  P(node) =0.01604621
##     class counts:     0    25     0     0
##    probabilities: 0.000 1.000 0.000 0.000 
## 
## Node number 377: 15 observations
##   predicted class=R  expected loss=0.2666667  P(node) =0.009627728
##     class counts:     0     4     0    11
##    probabilities: 0.000 0.267 0.000 0.733 
## 
## Node number 382: 11 observations
##   predicted class=R  expected loss=0.5454545  P(node) =0.007060334
##     class counts:     4     2     0     5
##    probabilities: 0.364 0.182 0.000 0.455 
## 
## Node number 383: 264 observations,    complexity param=0.001513841
##   predicted class=R  expected loss=0.04545455  P(node) =0.169448
##     class counts:     0    12     0   252
##    probabilities: 0.000 0.045 0.000 0.955 
##   left son=766 (32 obs) right son=767 (232 obs)
##   Primary splits:
##       yedge < 6.5  to the right, improve=3.047022, (0 missing)
##       onpix < 6.5  to the right, improve=1.810730, (0 missing)
##       xedge < 4.5  to the right, improve=1.778004, (0 missing)
##       ybox  < 8.5  to the right, improve=1.609474, (0 missing)
##       xbar  < 8.5  to the right, improve=1.540684, (0 missing)
##   Surrogate splits:
##       xbox  < 6.5  to the right, agree=0.894, adj=0.125, (0 split)
##       xedge < 6.5  to the right, agree=0.894, adj=0.125, (0 split)
##       ybox  < 11.5 to the right, agree=0.890, adj=0.094, (0 split)
##       onpix < 7.5  to the right, agree=0.886, adj=0.062, (0 split)
##       xybar < 5.5  to the left,  agree=0.886, adj=0.062, (0 split)
## 
## Node number 766: 32 observations,    complexity param=0.001513841
##   predicted class=R  expected loss=0.25  P(node) =0.02053915
##     class counts:     0     8     0    24
##    probabilities: 0.000 0.250 0.000 0.750 
##   left son=1532 (20 obs) right son=1533 (12 obs)
##   Primary splits:
##       y2bar     < 3.5  to the right, improve=2.4000000, (0 missing)
##       xy2bar    < 7.5  to the left,  improve=2.2500000, (0 missing)
##       xedgeycor < 6.5  to the right, improve=1.2705880, (0 missing)
##       ybox      < 7.5  to the right, improve=1.1200000, (0 missing)
##       xybar     < 8.5  to the right, improve=0.9468599, (0 missing)
##   Surrogate splits:
##       ybox   < 7.5  to the right, agree=0.844, adj=0.583, (0 split)
##       height < 5.5  to the right, agree=0.750, adj=0.333, (0 split)
##       xbar   < 7.5  to the left,  agree=0.750, adj=0.333, (0 split)
##       ybar   < 6.5  to the right, agree=0.750, adj=0.333, (0 split)
##       x2bar  < 5.5  to the left,  agree=0.750, adj=0.333, (0 split)
## 
## Node number 767: 232 observations
##   predicted class=R  expected loss=0.01724138  P(node) =0.1489089
##     class counts:     0     4     0   228
##    probabilities: 0.000 0.017 0.000 0.983 
## 
## Node number 1532: 20 observations,    complexity param=0.001513841
##   predicted class=R  expected loss=0.4  P(node) =0.01283697
##     class counts:     0     8     0    12
##    probabilities: 0.000 0.400 0.000 0.600 
##   left son=3064 (7 obs) right son=3065 (13 obs)
##   Primary splits:
##       xy2bar < 7.5  to the left,  improve=7.753846, (0 missing)
##       xbox   < 5.5  to the left,  improve=3.600000, (0 missing)
##       ybox   < 9.5  to the left,  improve=3.600000, (0 missing)
##       width  < 6.5  to the left,  improve=2.327273, (0 missing)
##       height < 6.5  to the left,  improve=2.327273, (0 missing)
##   Surrogate splits:
##       xedge < 2.5  to the left,  agree=0.80, adj=0.429, (0 split)
##       xbox  < 5.5  to the left,  agree=0.75, adj=0.286, (0 split)
##       ybox  < 9.5  to the left,  agree=0.75, adj=0.286, (0 split)
##       width < 5.5  to the left,  agree=0.75, adj=0.286, (0 split)
##       xbar  < 7.5  to the right, agree=0.75, adj=0.286, (0 split)
## 
## Node number 1533: 12 observations
##   predicted class=R  expected loss=0  P(node) =0.007702182
##     class counts:     0     0     0    12
##    probabilities: 0.000 0.000 0.000 1.000 
## 
## Node number 3064: 7 observations
##   predicted class=B  expected loss=0  P(node) =0.00449294
##     class counts:     0     7     0     0
##    probabilities: 0.000 1.000 0.000 0.000 
## 
## Node number 3065: 13 observations
##   predicted class=R  expected loss=0.07692308  P(node) =0.008344031
##     class counts:     0     1     0    12
##    probabilities: 0.000 0.077 0.000 0.923 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##    1) root 1558 1156 P (0.252888318 0.245827985 0.258023107 0.243260591)  
##      2) xedgeycor< 8.5 1088  706 A (0.351102941 0.310661765 0.011948529 0.326286765)  
##        4) ybar< 5.5 344   15 A (0.956395349 0.026162791 0.008720930 0.008720930)  
##          8) y2bar< 4.5 328    1 A (0.996951220 0.000000000 0.003048780 0.000000000) *
##          9) y2bar>=4.5 16    7 B (0.125000000 0.562500000 0.125000000 0.187500000) *
##        5) ybar>=5.5 744  392 R (0.071236559 0.442204301 0.013440860 0.473118280)  
##         10) xedgeycor>=7.5 342   75 B (0.040935673 0.780701754 0.029239766 0.149122807)  
##           20) xy2bar< 7.5 283   24 B (0.010600707 0.915194346 0.028268551 0.045936396)  
##             40) xedge< 4.5 265   12 B (0.003773585 0.954716981 0.018867925 0.022641509)  
##               80) xybar< 10.5 254    7 B (0.003937008 0.972440945 0.000000000 0.023622047)  
##                160) xedge< 2.5 206    0 B (0.000000000 1.000000000 0.000000000 0.000000000) *
##                161) xedge>=2.5 48    7 B (0.020833333 0.854166667 0.000000000 0.125000000)  
##                  322) xbar>=6.5 39    2 B (0.025641026 0.948717949 0.000000000 0.025641026) *
##                  323) xbar< 6.5 9    4 R (0.000000000 0.444444444 0.000000000 0.555555556) *
##               81) xybar>=10.5 11    5 B (0.000000000 0.545454545 0.454545455 0.000000000) *
##             41) xedge>=4.5 18   11 R (0.111111111 0.333333333 0.166666667 0.388888889) *
##           21) xy2bar>=7.5 59   21 R (0.186440678 0.135593220 0.033898305 0.644067797)  
##             42) yedgexcor< 5.5 12    3 A (0.750000000 0.083333333 0.000000000 0.166666667) *
##             43) yedgexcor>=5.5 47   11 R (0.042553191 0.148936170 0.042553191 0.765957447)  
##               86) xbar>=8.5 8    3 B (0.125000000 0.625000000 0.000000000 0.250000000) *
##               87) xbar< 8.5 39    5 R (0.025641026 0.051282051 0.051282051 0.871794872) *
##         11) xedgeycor< 7.5 402  101 R (0.097014925 0.154228856 0.000000000 0.748756219)  
##           22) yedge< 2.5 26    1 A (0.961538462 0.000000000 0.000000000 0.038461538) *
##           23) yedge>=2.5 376   76 R (0.037234043 0.164893617 0.000000000 0.797872340)  
##             46) yedge>=7.5 26    8 B (0.153846154 0.692307692 0.000000000 0.153846154)  
##               92) xybar>=8.5 16    0 B (0.000000000 1.000000000 0.000000000 0.000000000) *
##               93) xybar< 8.5 10    6 A (0.400000000 0.200000000 0.000000000 0.400000000) *
##             47) yedge< 7.5 350   54 R (0.028571429 0.125714286 0.000000000 0.845714286)  
##               94) xybar>=9.5 68   29 R (0.000000000 0.426470588 0.000000000 0.573529412)  
##                188) yedge>=4.5 40   11 B (0.000000000 0.725000000 0.000000000 0.275000000)  
##                  376) x2ybar>=3.5 25    0 B (0.000000000 1.000000000 0.000000000 0.000000000) *
##                  377) x2ybar< 3.5 15    4 R (0.000000000 0.266666667 0.000000000 0.733333333) *
##                189) yedge< 4.5 28    0 R (0.000000000 0.000000000 0.000000000 1.000000000) *
##               95) xybar< 9.5 282   25 R (0.035460993 0.053191489 0.000000000 0.911347518)  
##                190) x2ybar>=6.5 7    1 A (0.857142857 0.142857143 0.000000000 0.000000000) *
##                191) x2ybar< 6.5 275   18 R (0.014545455 0.050909091 0.000000000 0.934545455)  
##                  382) yedgexcor< 6.5 11    6 R (0.363636364 0.181818182 0.000000000 0.454545455) *
##                  383) yedgexcor>=6.5 264   12 R (0.000000000 0.045454545 0.000000000 0.954545455)  
##                    766) yedge>=6.5 32    8 R (0.000000000 0.250000000 0.000000000 0.750000000)  
##                     1532) y2bar>=3.5 20    8 R (0.000000000 0.400000000 0.000000000 0.600000000)  
##                       3064) xy2bar< 7.5 7    0 B (0.000000000 1.000000000 0.000000000 0.000000000) *
##                       3065) xy2bar>=7.5 13    1 R (0.000000000 0.076923077 0.000000000 0.923076923) *
##                     1533) y2bar< 3.5 12    0 R (0.000000000 0.000000000 0.000000000 1.000000000) *
##                    767) yedge< 6.5 232    4 R (0.000000000 0.017241379 0.000000000 0.982758621) *
##      3) xedgeycor>=8.5 470   81 P (0.025531915 0.095744681 0.827659574 0.051063830)  
##        6) xybar< 7.5 91   51 B (0.109890110 0.439560440 0.219780220 0.230769231)  
##         12) x2bar>=3.5 55   17 B (0.018181818 0.690909091 0.236363636 0.054545455)  
##           24) xedgeycor< 10.5 46   10 B (0.021739130 0.782608696 0.130434783 0.065217391) *
##           25) xedgeycor>=10.5 9    2 P (0.000000000 0.222222222 0.777777778 0.000000000) *
##         13) x2bar< 3.5 36   18 R (0.250000000 0.055555556 0.194444444 0.500000000)  
##           26) x2ybar>=6.5 16    7 A (0.562500000 0.000000000 0.437500000 0.000000000) *
##           27) x2ybar< 6.5 20    2 R (0.000000000 0.100000000 0.000000000 0.900000000) *
##        7) xybar>=7.5 379   10 P (0.005277045 0.013192612 0.973614776 0.007915567) *
##          Prediction
## Reference   A   B   P   R
##         A 380   5   2   7
##         B   4 347   7  25
##         P   8  13 376   5
##         R   7   9   3 360
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9390244      0.9186971      0.9259714      0.9503903      0.2580231 
## AccuracyPValue  McnemarPValue 
##      0.0000000      0.0352118 
##          Prediction
## Reference   A   B   P   R
##         A 375   6   2  12
##         B   5 326  21  31
##         P   8  16 367  10
##         R  10  29   6 334
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8998716      0.8664787      0.8838837      0.9143333      0.2573813 
## AccuracyPValue  McnemarPValue 
##      0.0000000      0.4676595 
##           model_id model_method
## 1 All.X.cp.0.rpart        rpart
##                                                                                                                   feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                      0.567                  0.09
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.9390244             0.9259714             0.9503903
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1     0.9186971        0.8998716             0.8838837
##   max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.9143333     0.8664787
## [1] "iterating over method:rf"
## [1] "fitting model: All.X.rf"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm"
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

![](Letter_Recognition_files/figure-html/fit.models_0-11.png) 

```
## + : mtry= 2 
## - : mtry= 2 
## + : mtry= 9 
## - : mtry= 9 
## + : mtry=17 
## - : mtry=17 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## mtry
```

![](Letter_Recognition_files/figure-html/fit.models_0-12.png) ![](Letter_Recognition_files/figure-html/fit.models_0-13.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        2500   -none-     numeric  
## confusion         20   -none-     numeric  
## votes           6232   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            4   -none-     character
## importance        17   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y               1558   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames            17   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          4   -none-     character
##          Prediction
## Reference   A   B   P   R
##         A 394   0   0   0
##         B   0 383   0   0
##         P   0   0 402   0
##         R   0   0   0 379
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      1.0000000      1.0000000      0.9976351      1.0000000      0.2580231 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference   A   B   P   R
##         A 389   0   4   2
##         B   0 381   0   2
##         P   0   6 391   4
##         R   0   9   0 370
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9826701      0.9768917      0.9748854      0.9885492      0.2573813 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##   model_id model_method
## 1 All.X.rf           rf
##                                                                                                                           feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      9.437                 2.024
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.9826701             0.9976351                     1
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1     0.9768909        0.9826701             0.9748854
##   max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.9885492     0.9768917
```

```r
# Simplify a model
# fit_df <- glb_trnent_df; glb_mdl <- step(<complex>_mdl)

print(glb_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6  Interact.High.cor.y.rpart            rpart
## 7            Low.cor.X.rpart            rpart
## 8                All.X.rpart            rpart
## 9           All.X.cp.0.rpart            rpart
## 10                  All.X.rf               rf
##                                                                                                                            feats
## 1                                                                                                                         .rnorm
## 2                                                                                                                         .rnorm
## 3                                                                                                                           ybar
## 4                                                                                                                           ybar
## 5                                                                                                                           ybar
## 6                                                                                           ybar, xedgeycor, width, height, ybox
## 7                                          ybar, x2bar, x2ybar, y2bar, yedgexcor, yedge, xbox, onpix, xybar, xedge, xy2bar, xbar
## 8          xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 9          xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 10 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##    max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1                0                      0.398                 0.002
## 2                0                      0.224                 0.001
## 3                0                      0.565                 0.025
## 4                0                      0.460                 0.023
## 5                3                      1.041                 0.025
## 6                3                      1.238                 0.046
## 7                3                      1.403                 0.081
## 8                3                      1.239                 0.096
## 9                0                      0.567                 0.090
## 10               3                      9.437                 2.024
##    max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1         0.2580231             0.2364483             0.2805136
## 2         0.2458280             0.2246218             0.2679962
## 3         0.2580231             0.2364483             0.2805136
## 4         0.6707317             0.6467703             0.6940479
## 5         0.6598241             0.6330850             0.6808266
## 6         0.7721086             0.6630906             0.7097592
## 7         0.7291809             0.6474225             0.6946769
## 8         0.7721086             0.6630906             0.7097592
## 9         0.9390244             0.9259714             0.9503903
## 10        0.9826701             0.9976351             1.0000000
##    max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1    0.000000000        0.2573813             0.2358253
## 2   -0.005794174        0.2483954             0.2271097
## 3    0.000000000        0.2573813             0.2358253
## 4    0.560539414        0.6662388             0.6422061
## 5    0.545734199        0.6508344             0.6265756
## 6    0.696187593        0.6848524             0.6611305
## 7    0.639086619        0.6655969             0.6415543
## 8    0.696187593        0.6848524             0.6611305
## 9    0.918697133        0.8998716             0.8838837
## 10   0.976890902        0.9826701             0.9748854
##    max.AccuracyUpper.OOB max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1              0.2798554   0.000000000                 NA              NA
## 2              0.2706334  -0.002275479                 NA              NA
## 3              0.2798554   0.000000000                 NA              NA
## 4              0.6896432   0.554721602                 NA              NA
## 5              0.6745233   0.533898743         0.01444694      0.01906184
## 6              0.7078755   0.579853246         0.09299090      0.12361354
## 7              0.6890137   0.554832242         0.05790668      0.07708770
## 8              0.7078755   0.579853246         0.09299090      0.12361354
## 9              0.9143333   0.866478712                 NA              NA
## 10             0.9885492   0.976891689                 NA              NA
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,                              
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##          chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8  fit.models                5                0  10.339
## elapsed9  fit.models                5                1  46.410
```

## Step `5`: fit models

```r
if (!is.null(glb_model_metric_smmry)) {
    stats_df <- glb_models_df[, "model_id", FALSE]

    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_trnent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "fit",
        						glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_newent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "OOB",
            					glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
#     tmp_models_df <- orderBy(~model_id, glb_models_df)
#     rownames(tmp_models_df) <- seq(1, nrow(tmp_models_df))
#     all.equal(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr"),
#               subset(stats_df, model_id != "Random.myrandom_classfr"))
#     print(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])
#     print(subset(stats_df, model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])

    print("Merging following data into glb_models_df:")
    print(stats_mrg_df <- stats_df[, c(1, grep(glb_model_metric, names(stats_df)))])
    print(tmp_models_df <- orderBy(~model_id, glb_models_df[, c("model_id", grep(glb_model_metric, names(stats_df), value=TRUE))]))

    tmp2_models_df <- glb_models_df[, c("model_id", setdiff(names(glb_models_df), grep(glb_model_metric, names(stats_df), value=TRUE)))]
    tmp3_models_df <- merge(tmp2_models_df, stats_mrg_df, all.x=TRUE, sort=FALSE)
    print(tmp3_models_df)
    print(names(tmp3_models_df))
    print(glb_models_df <- subset(tmp3_models_df, select=-model_id.1))
}

plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6  Interact.High.cor.y.rpart            rpart
## 7            Low.cor.X.rpart            rpart
## 8                All.X.rpart            rpart
## 9           All.X.cp.0.rpart            rpart
## 10                  All.X.rf               rf
##                                                                                                                            feats
## 1                                                                                                                         .rnorm
## 2                                                                                                                         .rnorm
## 3                                                                                                                           ybar
## 4                                                                                                                           ybar
## 5                                                                                                                           ybar
## 6                                                                                           ybar, xedgeycor, width, height, ybox
## 7                                          ybar, x2bar, x2ybar, y2bar, yedgexcor, yedge, xbox, onpix, xybar, xedge, xy2bar, xbar
## 8          xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 9          xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 10 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##    max.nTuningRuns max.Accuracy.fit max.Kappa.fit max.Accuracy.OOB
## 1                0        0.2580231   0.000000000        0.2573813
## 2                0        0.2458280  -0.005794174        0.2483954
## 3                0        0.2580231   0.000000000        0.2573813
## 4                0        0.6707317   0.560539414        0.6662388
## 5                3        0.6598241   0.545734199        0.6508344
## 6                3        0.7721086   0.696187593        0.6848524
## 7                3        0.7291809   0.639086619        0.6655969
## 8                3        0.7721086   0.696187593        0.6848524
## 9                0        0.9390244   0.918697133        0.8998716
## 10               3        0.9826701   0.976890902        0.9826701
##    max.Kappa.OOB inv.elapsedtime.everything inv.elapsedtime.final
## 1    0.000000000                  2.5125628           500.0000000
## 2   -0.002275479                  4.4642857          1000.0000000
## 3    0.000000000                  1.7699115            40.0000000
## 4    0.554721602                  2.1739130            43.4782609
## 5    0.533898743                  0.9606148            40.0000000
## 6    0.579853246                  0.8077544            21.7391304
## 7    0.554832242                  0.7127584            12.3456790
## 8    0.579853246                  0.8071025            10.4166667
## 9    0.866478712                  1.7636684            11.1111111
## 10   0.976891689                  0.1059659             0.4940711
```

```r
print(myplot_radar(radar_inp_df=plt_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 10. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 32 rows containing missing values (geom_point).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 10. Consider specifying shapes manually. if you must have them.
```

![](Letter_Recognition_files/figure-html/fit.models_1-1.png) 

```r
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(model_id %in% grep("random|MFO", plt_models_df$model_id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## Warning: max.AccuracyUpper.fit already exists in glb_models_df
```

```
## [1] "var:max.KappaSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "model_id", FALSE]
pltCI_models_df <- glb_models_df[, "model_id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="model_id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="model_id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
#print(mltdCI_models_df)
# castCI_models_df <- dcast(mltdCI_models_df, value ~ type, fun.aggregate=sum)
# print(castCI_models_df)
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data, sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("model_id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("model_id", "model_method")], all.x=TRUE)

# print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="data") + 
#         geom_errorbar(data=mrgdCI_models_df, 
#             mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
#           facet_grid(label ~ data, scales="free") + 
#           theme(axis.text.x = element_text(angle = 45,vjust = 1)))
# mltd_models_df <- orderBy(~ value +variable +data +label + model_method + model_id, 
#                           mltd_models_df)
print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="model_method") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
```

```
## Warning: Stacking not well defined when ymin != 0
```

```
## Warning: Stacking not well defined when ymin != 0
```

![](Letter_Recognition_files/figure-html/fit.models_1-2.png) 

```r
model_evl_terms <- c(NULL)
for (metric in glb_model_evl_criteria)
    model_evl_terms <- c(model_evl_terms, 
                    ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse=" "))
print(tmp_models_df <- orderBy(model_sel_frmla, glb_models_df)[, c("model_id", glb_model_evl_criteria)])
```

```
##                     model_id max.Accuracy.OOB max.Kappa.OOB
## 10                  All.X.rf        0.9826701   0.976891689
## 9           All.X.cp.0.rpart        0.8998716   0.866478712
## 6  Interact.High.cor.y.rpart        0.6848524   0.579853246
## 8                All.X.rpart        0.6848524   0.579853246
## 4  Max.cor.Y.cv.0.cp.0.rpart        0.6662388   0.554721602
## 7            Low.cor.X.rpart        0.6655969   0.554832242
## 5            Max.cor.Y.rpart        0.6508344   0.533898743
## 1          MFO.myMFO_classfr        0.2573813   0.000000000
## 3       Max.cor.Y.cv.0.rpart        0.2573813   0.000000000
## 2    Random.myrandom_classfr        0.2483954  -0.002275479
```

```r
print(myplot_radar(radar_inp_df=tmp_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 10. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 12 rows containing missing values (geom_point).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 10. Consider specifying shapes manually. if you must have them.
```

![](Letter_Recognition_files/figure-html/fit.models_1-3.png) 

```r
print("Metrics used for model selection:"); print(model_sel_frmla)
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB
```

```r
print(sprintf("Best model id: %s", tmp_models_df[1, "model_id"]))
```

```
## [1] "Best model id: All.X.rf"
```

```r
if (is.null(glb_sel_mdl_id)) 
    { glb_sel_mdl_id <- tmp_models_df[1, "model_id"] } else 
        print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
    
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

![](Letter_Recognition_files/figure-html/fit.models_1-4.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        2500   -none-     numeric  
## confusion         20   -none-     numeric  
## votes           6232   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            4   -none-     character
## importance        17   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y               1558   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames            17   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          4   -none-     character
```

```
## [1] TRUE
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](Letter_Recognition_files/figure-html/fit.models_1-5.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed9             fit.models                5                1  46.410
## elapsed10 fit.data.training.all                6                0  53.674
```

## Step `6`: fit.data.training.all

```r
if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_sel_mdl
} else {    
    print(mdl_feats_df <- myextract_mdl_feats(sel_mdl=glb_sel_mdl, entity_df=glb_trnent_df))
    
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the model_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
        
    # Sync with parameters in mydsutils.R
    ret_lst <- myfit_mdl(model_id="Final", model_method=model_method,
                            indep_vars_vctr=mdl_feats_df$id, model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out, 
                            fit_df=glb_trnent_df, OOB_df=NULL,
                         # Automate from here
                         #  Issues if glb_sel_mdl$method == "rf" b/c trainControl is "oob"; not "cv"
                         n_cv_folds=glb_n_cv_folds, tune_models_df=NULL,
                            model_loss_mtrx=glb_model_metric_terms,
                            model_summaryFunction=glb_sel_mdl$control$summaryFunction,
                            model_metric=glb_sel_mdl$metric,
                            model_metric_maximize=glb_sel_mdl$maximize)
    glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]] 
    glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "model_id"]
}
```

```
##            importance        id fit.feat
## xedgeycor 100.0000000 xedgeycor     TRUE
## xy2bar     89.3798544    xy2bar     TRUE
## ybar       84.4518694      ybar     TRUE
## x2ybar     67.2579886    x2ybar     TRUE
## yedge      55.1439421     yedge     TRUE
## y2bar      46.1842883     y2bar     TRUE
## yedgexcor  40.5480335 yedgexcor     TRUE
## xybar      35.9840226     xybar     TRUE
## xedge      31.4944591     xedge     TRUE
## x2bar      28.4852495     x2bar     TRUE
## xbar       19.0667824      xbar     TRUE
## onpix       5.8408785     onpix     TRUE
## ybox        2.0320898      ybox     TRUE
## height      1.6484989    height     TRUE
## xbox        1.1858412      xbox     TRUE
## .rnorm      0.8184787    .rnorm     TRUE
## width       0.0000000     width     TRUE
## [1] "fitting model: Final.rf"
## [1] "    indep_vars: xedgeycor, xy2bar, ybar, x2ybar, yedge, y2bar, yedgexcor, xybar, xedge, x2bar, xbar, onpix, ybox, height, xbox, .rnorm, width"
## + : mtry= 2 
## - : mtry= 2 
## + : mtry= 9 
## - : mtry= 9 
## + : mtry=17 
## - : mtry=17 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

```
## Warning in myfit_mdl(model_id = "Final", model_method = model_method,
## indep_vars_vctr = mdl_feats_df$id, : model's bestTune found at an extreme
## of tuneGrid for parameter: mtry
```

![](Letter_Recognition_files/figure-html/fit.data.training.all_0-1.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        2500   -none-     numeric  
## confusion         20   -none-     numeric  
## votes           6232   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            4   -none-     character
## importance        17   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y               1558   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames            17   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          4   -none-     character
##          Prediction
## Reference   A   B   P   R
##         A 394   0   0   0
##         B   0 383   0   0
##         P   0   0 402   0
##         R   0   0   0 379
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      1.0000000      1.0000000      0.9976351      1.0000000      0.2580231 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```
## Warning in mypredict_mdl(mdl, df = fit_df, rsp_var, rsp_var_out,
## model_id_method, : Expecting 1 metric: Accuracy; recd: Accuracy, Kappa;
## retaining Accuracy only
```

![](Letter_Recognition_files/figure-html/fit.data.training.all_0-2.png) 

```
##   model_id model_method
## 1 Final.rf           rf
##                                                                                                                           feats
## 1 xedgeycor, xy2bar, ybar, x2ybar, yedge, y2bar, yedgexcor, xybar, xedge, x2bar, xbar, onpix, ybox, height, xbox, .rnorm, width
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      8.341                 1.924
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.9807445             0.9976351                     1
##   max.Kappa.fit
## 1     0.9743239
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed10 fit.data.training.all                6                0  53.674
## elapsed11 fit.data.training.all                6                1  64.375
```


```r
glb_rsp_var_out <- paste0(glb_rsp_var_out, tail(names(glb_models_lst), 1))

# Used again in predict.data.new chunk
glb_get_predictions <- function(df) {
    if (glb_is_regression) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
        print(myplot_scatter(df, glb_rsp_var, glb_rsp_var_out, 
                             smooth=TRUE))
        df[, paste0(glb_rsp_var_out, ".err")] <- 
            abs(df[, glb_rsp_var_out] - df[, glb_rsp_var])
        print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                           df)))                             
    }

    if (glb_is_classification && glb_is_binomial) {
        # incorporate glb_clf_proba_threshold
        #   shd it only be for glb_fin_mdl or for earlier models ?
        if ((prob_threshold <- 
                glb_models_df[glb_models_df$model_id == glb_fin_mdl_id, "opt.prob.threshold.fit"]) != 
            glb_models_df[glb_models_df$model_id == glb_sel_mdl_id, "opt.prob.threshold.fit"])
            stop("user specification for probability threshold required")
        
        df[, paste0(glb_rsp_var_out, ".prob")] <- 
            predict(glb_fin_mdl, newdata=df, type="prob")[, 2]
        df[, glb_rsp_var_out] <- 
    			factor(levels(df[, glb_rsp_var])[
    				(df[, paste0(glb_rsp_var_out, ".prob")] >=
    					prob_threshold) * 1 + 1], levels(df[, glb_rsp_var]))
    
        # prediction stats already reported by myfit_mdl ???
    }    
    
    if (glb_is_classification && !glb_is_binomial) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
    }

    return(df)
}    
glb_trnent_df <- glb_get_predictions(glb_trnent_df)

print(glb_feats_df <- mymerge_feats_importance(feats_df=glb_feats_df, sel_mdl=glb_fin_mdl, 
                                               entity_df=glb_trnent_df))
```

```
##           id       cor.y exclude.as.feat  cor.y.abs cor.low importance
## 10 xedgeycor  0.27453618               0 0.27453618       0 100.000000
## 14      ybar  0.67198759               0 0.67198759       1  81.764532
## 11    xy2bar -0.27957030               0 0.27957030       1  81.567199
## 6     x2ybar  0.38607514               0 0.38607514       1  66.452551
## 16     yedge  0.24976884               0 0.24976884       1  55.445165
## 13     y2bar  0.33859131               0 0.33859131       1  46.090470
## 17 yedgexcor  0.31367199               0 0.31367199       1  38.683780
## 12     xybar  0.12073750               0 0.12073750       1  34.339849
## 9      xedge  0.11786463               0 0.11786463       1  33.721463
## 5      x2bar  0.41019605               0 0.41019605       1  30.680513
## 7       xbar -0.41476375               0 0.41476375       1  18.196502
## 3      onpix  0.16721154               0 0.16721154       1   5.242872
## 15      ybox  0.03690669               0 0.03690669       0   2.144477
## 8       xbox  0.16830409               0 0.16830409       1   1.949979
## 1     .rnorm -0.03047401               0 0.03047401       1   1.695361
## 2     height  0.04565534               0 0.04565534       0   1.555729
## 4      width  0.04909820               0 0.04909820       0   0.000000
```

```r
# Used again in predict.data.new chunk
glb_analytics_diag_plots <- function(obs_df) {
    for (var in subset(glb_feats_df, !is.na(importance))$id) {
        plot_df <- melt(obs_df, id.vars=var, 
                        measure.vars=c(glb_rsp_var, glb_rsp_var_out))
#         if (var == "<feat_name>") print(myplot_scatter(plot_df, var, "value", 
#                                              facet_colcol_name="variable") + 
#                       geom_vline(xintercept=<divider_val>, linetype="dotted")) else     
            print(myplot_scatter(plot_df, var, "value", colorcol_name="variable",
                                 facet_colcol_name="variable", jitter=TRUE) + 
                      guides(color=FALSE))
    }
    
    if (glb_is_regression) {
        plot_vars_df <- subset(glb_feats_df, importance > glb_feats_df[glb_feats_df$id == ".rnorm", "importance"])
        print(myplot_prediction_regression(df=obs_df, 
                    feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], ".rownames"), 
                                           feat_y=plot_vars_df$id[1],
                    rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                    id_vars=glb_id_vars)
#               + facet_wrap(reformulate(plot_vars_df$id[2])) # if [1 or 2] is a factor                                                         
#               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
              )
    }    
    
    if (glb_is_classification) {
        if (nrow(plot_vars_df <- subset(glb_feats_df, !is.na(importance))) == 0)
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df=obs_df, 
                feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], 
                              ".rownames"),
                                               feat_y=plot_vars_df$id[1],
                     rsp_var=glb_rsp_var, 
                     rsp_var_out=glb_rsp_var_out, 
                     id_vars=glb_id_vars)
#               + geom_hline(yintercept=<divider_val>, linetype = "dotted")
                )
    }    
}
glb_analytics_diag_plots(obs_df=glb_trnent_df)
```

![](Letter_Recognition_files/figure-html/fit.data.training.all_1-1.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-2.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-3.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-4.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-5.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-6.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-7.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-8.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-9.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-10.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-11.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-12.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-13.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-14.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-15.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-16.png) ![](Letter_Recognition_files/figure-html/fit.data.training.all_1-17.png) 

```
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 238       A    7   12     6      6     3    9    0     3     2     9
## 737       P    5    5     7      7     8    8    7     4     2     7
## 1380      P    5    9     5      6     2    4   15     8     1    12
## 3057      A    7   14     7      8     6   10    3     5     2    10
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor     .rnorm letter.fctr
## 238       4     12     3         5     4         6  0.4917073           A
## 737       8      7     7        12     5         7 -1.0490578           P
## 1380      6      2     0         9     4         8 -1.7707352           P
## 3057      5     12     7         1     6        11 -2.2933418           A
##      letter.fctr.predict.Final.rf letter.fctr.predict.Final.rf.accurate
## 238                             A                                  TRUE
## 737                             P                                  TRUE
## 1380                            P                                  TRUE
## 3057                            A                                  TRUE
##      .label
## 238    .238
## 737    .737
## 1380  .1380
## 3057  .3057
```

![](Letter_Recognition_files/figure-html/fit.data.training.all_1-18.png) 

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](Letter_Recognition_files/figure-html/fit.data.training.all_1-19.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="predict.data.new", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed11 fit.data.training.all                6                1  64.375
## elapsed12      predict.data.new                7                0  77.582
```

## Step `7`: predict data.new

```r
# Compute final model predictions
glb_newent_df <- glb_get_predictions(glb_newent_df)
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

![](Letter_Recognition_files/figure-html/predict.data.new-1.png) ![](Letter_Recognition_files/figure-html/predict.data.new-2.png) ![](Letter_Recognition_files/figure-html/predict.data.new-3.png) ![](Letter_Recognition_files/figure-html/predict.data.new-4.png) ![](Letter_Recognition_files/figure-html/predict.data.new-5.png) ![](Letter_Recognition_files/figure-html/predict.data.new-6.png) ![](Letter_Recognition_files/figure-html/predict.data.new-7.png) ![](Letter_Recognition_files/figure-html/predict.data.new-8.png) ![](Letter_Recognition_files/figure-html/predict.data.new-9.png) ![](Letter_Recognition_files/figure-html/predict.data.new-10.png) ![](Letter_Recognition_files/figure-html/predict.data.new-11.png) ![](Letter_Recognition_files/figure-html/predict.data.new-12.png) ![](Letter_Recognition_files/figure-html/predict.data.new-13.png) ![](Letter_Recognition_files/figure-html/predict.data.new-14.png) ![](Letter_Recognition_files/figure-html/predict.data.new-15.png) ![](Letter_Recognition_files/figure-html/predict.data.new-16.png) ![](Letter_Recognition_files/figure-html/predict.data.new-17.png) 

```
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 314       A    5   12     5      6     3   14    0     4     2    12
## 364       A    8   14     8      8     6    9    4     5     4    10
## 877       P    6    9     9      7     4    5   15     6     2    12
## 2214      P    6    6     8      8     9    6    7     5     3     8
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor      .rnorm letter.fctr
## 314       3     11     2         4     2        10  0.27395391           A
## 364       6     12     9         2     7        11 -0.07503107           A
## 877       5      1     0         9     4         7 -0.05228726           P
## 2214      7      6     7        13     6         9 -0.21365004           P
##      letter.fctr.predict.Final.rf letter.fctr.predict.Final.rf.accurate
## 314                             A                                  TRUE
## 364                             A                                  TRUE
## 877                             P                                  TRUE
## 2214                            P                                  TRUE
##      .label
## 314    .314
## 364    .364
## 877    .877
## 2214  .2214
```

![](Letter_Recognition_files/figure-html/predict.data.new-18.png) 

```r
tmp_replay_lst <- replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.new.prediction")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1 
## 6.0000 	 6 	 0 0 1 2
```

![](Letter_Recognition_files/figure-html/predict.data.new-19.png) 

```r
print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

![](Letter_Recognition_files/figure-html/predict.data.new-20.png) 

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                   chunk_label chunk_step_major chunk_step_minor elapsed
## 10                 fit.models                5                1  46.410
## 13           predict.data.new                7                0  77.582
## 12      fit.data.training.all                6                1  64.375
## 11      fit.data.training.all                6                0  53.674
## 9                  fit.models                5                0  10.339
## 6            extract_features                3                0   5.054
## 7             select_features                4                0   6.513
## 4         manage_missing_data                2                2   1.543
## 2                cleanse_data                2                0   0.548
## 5          encode_retype_data                2                3   1.986
## 8  remove_correlated_features                4                1   6.727
## 3       inspectORexplore.data                2                1   0.580
## 1                 import_data                1                0   0.002
##    elapsed_diff
## 10       36.071
## 13       13.207
## 12       10.701
## 11        7.264
## 9         3.612
## 6         3.068
## 7         1.459
## 4         0.963
## 2         0.546
## 5         0.443
## 8         0.214
## 3         0.032
## 1         0.000
```

```
## [1] "Total Elapsed Time: 77.582 secs"
```

![](Letter_Recognition_files/figure-html/print_sessionInfo-1.png) 

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.3 (Yosemite)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] tcltk     grid      stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] randomForest_4.6-10 rpart.plot_1.5.2    rpart_4.1-9        
##  [4] caret_6.0-41        lattice_0.20-31     sqldf_0.4-10       
##  [7] RSQLite_1.0.0       DBI_0.3.1           gsubfn_0.6-6       
## [10] proto_0.3-10        reshape2_1.4.1      plyr_1.8.1         
## [13] caTools_1.17.1      doBy_4.5-13         survival_2.38-1    
## [16] ggplot2_1.0.1      
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gtable_0.1.2       
## [16] gtools_3.4.1        htmltools_0.2.6     iterators_1.0.7    
## [19] knitr_1.9           labeling_0.3        lme4_1.1-7         
## [22] MASS_7.3-40         Matrix_1.2-0        mgcv_1.8-6         
## [25] minqa_1.2.4         munsell_0.4.2       nlme_3.1-120       
## [28] nloptr_1.0.4        nnet_7.3-9          parallel_3.1.3     
## [31] pbkrtest_0.4-2      quantreg_5.11       RColorBrewer_1.1-2 
## [34] Rcpp_0.11.5         rmarkdown_0.5.1     scales_0.2.4       
## [37] SparseM_1.6         splines_3.1.3       stringr_0.6.2      
## [40] tools_3.1.3         yaml_2.1.13
```
