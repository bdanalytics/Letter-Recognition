# UCI ML Repo::Letter Recognition: isB classification
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
glb_split_sample.seed <- 1000               # or any integer
glb_max_obs <- NULL # or any integer

glb_is_regression <- FALSE; glb_is_classification <- TRUE; glb_is_binomial <- TRUE

glb_rsp_var_raw <- "letter"

# for classification, the response variable has to be a factor
glb_rsp_var <- "isB"

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    relevel(factor(ifelse(raw == "B", "Y", "N")), as.factor(c("Y", "N")), ref="N")
    #as.factor(paste0("B", raw))
}
glb_map_rsp_raw_to_var(c("A", "B", "P", "R"))
```

```
## [1] N Y N N
## Levels: N Y
```

```r
glb_map_rsp_var_to_raw <- function(var) {
    as.numeric(var) - 1
    #as.numeric(var)
}
glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c("A", "B", "P", "R")))
```

```
## [1] 0 1 0 0
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

![](Letter_Recognition_B_files/figure-html/set_global_options-1.png) 

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
## elapsed import_data                1                0   0.003
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
##   letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 2      A    1    1     3      2     1    8    2     2     2     8      2
## 3      R    5    9     5      7     6    6   11     7     3     7      3
## 4      B    5    9     7      7    10    9    8     4     4     6      8
## 5      P    3    6     4      4     2    4   14     8     1    11      6
## 7      R    2    6     4      4     3    6    7     5     5     6      5
## 9      P    8   14     7      8     4    5   10     6     3    12      5
##   xy2bar xedge xedgeycor yedge yedgexcor
## 2      8     1         6     2         7
## 3      9     2         7     5        11
## 4      6     6        11     8         7
## 5      3     0        10     4         8
## 7      7     3         7     5         8
## 9      4     4        10     4         8
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 304       P    3    4     5      6     7    8    5     5     2     7
## 1693      R    5    7     7      5     4   10    7     3     6    10
## 1959      B    2    4     3      2     2    9    7     3     5    10
## 2386      P    3    2     4      3     2    5   10     4     4    10
## 2647      A    3    9     4      7     3    6    4     2     1     6
## 2802      B    5    7     7      5     5   10    6     3     7    11
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 304       6      7     6         9     6         9
## 1693      2      6     3         6     4        10
## 1959      5      7     2         8     4         9
## 2386      8      3     1        10     3         6
## 2647      1      8     2         6     2         7
## 2802      3      6     4         6     6        11
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 3106      A    3    7     4      5     2    7    4     2     0     6
## 3107      P    4    6     5      4     4    8    5     6     5     7
## 3108      P    4    9     6      6     4    7    9     3     4    12
## 3110      P    2    5     3      7     5    8    6     5     1     7
## 3112      A    2    3     3      1     1    6    2     2     1     5
## 3113      A    3    9     5      6     2    6    5     3     1     6
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 3106      2      8     2         7     1         7
## 3107      6      7     2         9     7         9
## 3108      5      4     2         9     3         8
## 3110      6      7     6         8     5         9
## 3112      2      8     1         6     1         7
## 3113      1      8     2         7     2         7
## 'data.frame':	1558 obs. of  17 variables:
##  $ letter   : chr  "A" "R" "B" "P" ...
##  $ xbox     : int  1 5 5 3 2 8 6 5 2 3 ...
##  $ ybox     : int  1 9 9 6 6 14 10 9 1 7 ...
##  $ width    : int  3 5 7 4 4 7 8 7 4 5 ...
##  $ height   : int  2 7 7 4 4 8 8 7 2 5 ...
##  $ onpix    : int  1 6 10 2 3 4 7 7 1 3 ...
##  $ xbar     : int  8 6 9 4 6 5 8 8 8 10 ...
##  $ ybar     : int  2 11 8 14 7 10 5 8 1 4 ...
##  $ x2bar    : int  2 7 4 8 5 6 7 3 2 1 ...
##  $ y2bar    : int  2 3 4 1 5 3 5 6 2 2 ...
##  $ xybar    : int  8 7 6 11 6 12 7 10 7 8 ...
##  $ x2ybar   : int  2 3 8 6 5 5 6 5 2 3 ...
##  $ xy2bar   : int  8 9 6 3 7 4 6 6 8 9 ...
##  $ xedge    : int  1 2 6 0 3 4 3 3 2 2 ...
##  $ xedgeycor: int  6 7 11 10 7 10 9 7 5 4 ...
##  $ yedge    : int  2 5 8 4 5 4 8 6 2 2 ...
##  $ yedgexcor: int  7 11 7 8 8 8 9 8 7 7 ...
##  - attr(*, "comment")= chr "glb_newent_df"
##    letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 1       B    4    2     5      4     4    8    7     6     6     7      6
## 6       R    8   10     8      6     6    7    7     3     5     8      4
## 8       A    3    7     5      5     3   12    2     3     2    10      2
## 11      A    3    8     5      6     3    9    2     2     3     8      2
## 12      R    6    9     5      4     3   10    6     5     5    10      2
## 13      B    3    3     3      4     3    7    7     5     5     7      6
##    xy2bar xedge xedgeycor yedge yedgexcor
## 1       6     2         8     7        10
## 6       8     6         6     7         7
## 8       9     2         6     3         8
## 11      8     2         6     3         7
## 12      8     6         6     4         9
## 13      6     5         8     5        10
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 33        A    3    5     5      7     2    7    7     3     1     6
## 633       P    5    8     8      6     4    6   12     7     2    11
## 1313      A    3    7     5      4     2    7    5     3     0     6
## 2010      B    4    7     5      5     5    8    6     6     7     6
## 2827      R    2    3     3      1     2    8    8     3     5     9
## 3026      B    4    8     6      6     6    8    7     4     5     7
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 33        0      8     3         7     1         8
## 633       5      2     1        10     4         9
## 1313      1      8     2         7     2         7
## 2010      6      6     2         8     7        10
## 2827      4      7     2         6     3        10
## 3026      6      7     4         8     6         9
##      letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar
## 3102      R    5    8     7      6     7    8    8     7     3     7
## 3109      A    6   10     7      6     3   12    0     4     1    11
## 3111      B    4    8     6      6     5    7    8     6     6    10
## 3114      R    2    3     3      2     2    7    7     5     5     7
## 3115      P    2    1     3      2     1    4   10     3     5    10
## 3116      A    4    9     6      6     2    9    5     3     1     8
##      x2ybar xy2bar xedge xedgeycor yedge yedgexcor
## 3102      4      6     5         7     7         8
## 3109      4     12     4         4     3        11
## 3111      6      6     3         8     7         8
## 3114      5      6     2         7     4         8
## 3115      8      5     0         9     3         7
## 3116      1      8     2         7     2         8
## 'data.frame':	1558 obs. of  17 variables:
##  $ letter   : chr  "B" "R" "A" "A" ...
##  $ xbox     : int  4 8 3 3 6 3 2 4 3 4 ...
##  $ ybox     : int  2 10 7 8 9 3 8 2 3 9 ...
##  $ width    : int  5 8 5 5 5 3 4 5 5 5 ...
##  $ height   : int  4 6 5 6 4 4 6 4 4 7 ...
##  $ onpix    : int  4 6 3 3 3 3 2 4 1 6 ...
##  $ xbar     : int  8 7 12 9 10 7 12 7 7 7 ...
##  $ ybar     : int  7 7 2 2 6 7 2 7 6 8 ...
##  $ x2bar    : int  6 3 3 2 5 5 4 5 3 9 ...
##  $ y2bar    : int  6 5 2 3 5 5 3 6 0 6 ...
##  $ xybar    : int  7 8 10 8 10 7 11 7 7 7 ...
##  $ x2ybar   : int  6 4 2 2 2 6 2 6 0 6 ...
##  $ xy2bar   : int  6 8 9 8 8 6 10 6 8 6 ...
##  $ xedge    : int  2 6 2 2 6 5 3 2 2 2 ...
##  $ xedgeycor: int  8 6 6 6 6 8 6 8 7 8 ...
##  $ yedge    : int  7 7 3 3 4 5 3 7 1 8 ...
##  $ yedgexcor: int  10 7 8 7 9 10 9 10 8 10 ...
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
## elapsed   import_data                1                0   0.003
## elapsed1 cleanse_data                2                0   0.561
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
## elapsed1          cleanse_data                2                0   0.561
## elapsed2 inspectORexplore.data                2                1   0.595
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
##  Min.   : 1.000   Min.   :-3.18310  
##  1st Qu.: 7.000   1st Qu.:-0.68688  
##  Median : 8.000   Median :-0.03311  
##  Mean   : 8.418   Mean   :-0.01089  
##  3rd Qu.:10.000   3rd Qu.: 0.67568  
##  Max.   :13.000   Max.   : 3.73914  
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
##     letter               xbox            ybox            width       
##  Length:1558        Min.   : 0.00   Min.   : 0.000   Min.   : 1.000  
##  Class :character   1st Qu.: 3.00   1st Qu.: 5.000   1st Qu.: 4.000  
##  Mode  :character   Median : 4.00   Median : 7.000   Median : 5.000  
##                     Mean   : 3.96   Mean   : 7.117   Mean   : 5.208  
##                     3rd Qu.: 5.00   3rd Qu.: 9.000   3rd Qu.: 6.000  
##                     Max.   :13.00   Max.   :15.000   Max.   :10.000  
##      height           onpix            xbar             ybar       
##  Min.   : 0.000   Min.   : 0.00   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 4.000   1st Qu.: 2.00   1st Qu.: 6.000   1st Qu.: 6.000  
##  Median : 6.000   Median : 4.00   Median : 7.000   Median : 7.000  
##  Mean   : 5.291   Mean   : 3.87   Mean   : 7.463   Mean   : 7.212  
##  3rd Qu.: 7.000   3rd Qu.: 5.00   3rd Qu.: 8.000   3rd Qu.: 9.000  
##  Max.   :11.000   Max.   :12.00   Max.   :14.000   Max.   :15.000  
##      x2bar            y2bar           xybar            x2ybar      
##  Min.   : 0.000   Min.   :0.000   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 3.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.: 3.000  
##  Median : 4.000   Median :4.000   Median : 8.000   Median : 5.000  
##  Mean   : 4.717   Mean   :3.892   Mean   : 8.495   Mean   : 4.501  
##  3rd Qu.: 6.000   3rd Qu.:5.000   3rd Qu.:10.000   3rd Qu.: 6.000  
##  Max.   :11.000   Max.   :8.000   Max.   :14.000   Max.   :10.000  
##      xy2bar           xedge         xedgeycor          yedge       
##  Min.   : 1.000   Min.   :0.000   Min.   : 1.000   Min.   : 1.000  
##  1st Qu.: 6.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.: 3.000  
##  Median : 7.000   Median :2.000   Median : 8.000   Median : 4.000  
##  Mean   : 6.718   Mean   :2.965   Mean   : 7.761   Mean   : 4.614  
##  3rd Qu.: 8.000   3rd Qu.:4.000   3rd Qu.: 9.000   3rd Qu.: 6.000  
##  Max.   :14.000   Max.   :9.000   Max.   :13.000   Max.   :12.000  
##    yedgexcor          .rnorm        
##  Min.   : 1.000   Min.   :-3.56179  
##  1st Qu.: 7.250   1st Qu.:-0.68418  
##  Median : 8.000   Median : 0.01442  
##  Mean   : 8.437   Mean   : 0.01240  
##  3rd Qu.:10.000   3rd Qu.: 0.67013  
##  Max.   :12.000   Max.   : 3.09890  
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
##     letter               xbox            ybox            width       
##  Length:1558        Min.   : 1.00   Min.   : 0.000   Min.   : 1.000  
##  Class :character   1st Qu.: 3.00   1st Qu.: 4.250   1st Qu.: 4.000  
##  Mode  :character   Median : 4.00   Median : 7.000   Median : 5.000  
##                     Mean   : 3.87   Mean   : 6.986   Mean   : 5.165  
##                     3rd Qu.: 5.00   3rd Qu.: 9.000   3rd Qu.: 6.000  
##                     Max.   :12.00   Max.   :15.000   Max.   :11.000  
##      height          onpix             xbar             ybar       
##  Min.   : 0.00   Min.   : 0.000   Min.   : 3.000   Min.   : 0.000  
##  1st Qu.: 4.00   1st Qu.: 2.000   1st Qu.: 6.000   1st Qu.: 6.000  
##  Median : 6.00   Median : 4.000   Median : 7.000   Median : 7.000  
##  Mean   : 5.26   Mean   : 3.867   Mean   : 7.474   Mean   : 7.182  
##  3rd Qu.: 7.00   3rd Qu.: 5.000   3rd Qu.: 9.000   3rd Qu.: 9.000  
##  Max.   :12.00   Max.   :12.000   Max.   :14.000   Max.   :15.000  
##      x2bar            y2bar           xybar            x2ybar     
##  Min.   : 1.000   Min.   :0.000   Min.   : 4.000   Min.   :0.000  
##  1st Qu.: 3.000   1st Qu.:2.000   1st Qu.: 7.000   1st Qu.:3.000  
##  Median : 4.000   Median :4.000   Median : 8.000   Median :5.000  
##  Mean   : 4.696   Mean   :3.915   Mean   : 8.488   Mean   :4.538  
##  3rd Qu.: 6.000   3rd Qu.:5.000   3rd Qu.:10.000   3rd Qu.:6.000  
##  Max.   :11.000   Max.   :8.000   Max.   :14.000   Max.   :9.000  
##      xy2bar           xedge          xedgeycor          yedge       
##  Min.   : 0.000   Min.   : 0.000   Min.   : 2.000   Min.   : 0.000  
##  1st Qu.: 6.000   1st Qu.: 2.000   1st Qu.: 7.000   1st Qu.: 3.000  
##  Median : 7.000   Median : 2.000   Median : 8.000   Median : 4.000  
##  Mean   : 6.704   Mean   : 2.861   Mean   : 7.765   Mean   : 4.587  
##  3rd Qu.: 8.000   3rd Qu.: 4.000   3rd Qu.: 9.000   3rd Qu.: 6.000  
##  Max.   :14.000   Max.   :10.000   Max.   :13.000   Max.   :11.000  
##    yedgexcor          .rnorm        
##  Min.   : 1.000   Min.   :-3.30491  
##  1st Qu.: 7.000   1st Qu.:-0.66110  
##  Median : 8.000   Median : 0.02347  
##  Mean   : 8.399   Mean   : 0.02235  
##  3rd Qu.:10.000   3rd Qu.: 0.72396  
##  Max.   :13.000   Max.   : 3.34667  
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

![](Letter_Recognition_B_files/figure-html/inspectORexplore_data-1.png) 

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
## elapsed2 inspectORexplore.data                2                1   0.595
## elapsed3   manage_missing_data                2                2   1.600
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
## elapsed3 manage_missing_data                2                2   1.600
## elapsed4  encode_retype_data                2                3   2.043
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
##   letter isB  .n
## 1      P   N 803
## 2      A   N 789
## 3      B   Y 766
## 4      R   N 758
```

![](Letter_Recognition_B_files/figure-html/encode_retype_data_1-1.png) 

```
##   letter isB  .n
## 1      P   N 402
## 2      A   N 394
## 3      B   Y 383
## 4      R   N 379
```

![](Letter_Recognition_B_files/figure-html/encode_retype_data_1-2.png) 

```
##   letter isB  .n
## 1      P   N 401
## 2      A   N 395
## 3      B   Y 383
## 4      R   N 379
```

![](Letter_Recognition_B_files/figure-html/encode_retype_data_1-3.png) 

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
## elapsed4 encode_retype_data                2                3   2.043
## elapsed5   extract_features                3                0   5.170
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

![](Letter_Recognition_B_files/figure-html/extract_features-1.png) 

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
## elapsed5 extract_features                3                0   5.170
## elapsed6  select_features                4                0   6.789
```

## Step `4`: select features

```r
print(glb_feats_df <- myselect_features(entity_df=glb_trnent_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                  id        cor.y exclude.as.feat   cor.y.abs
## yedge         yedge  0.561065858               0 0.561065858
## y2bar         y2bar  0.517624377               0 0.517624377
## x2ybar       x2ybar  0.286395882               0 0.286395882
## yedgexcor yedgexcor  0.219787666               0 0.219787666
## onpix         onpix  0.216051154               0 0.216051154
## x2bar         x2bar  0.147216267               0 0.147216267
## xybar         xybar -0.146207921               0 0.146207921
## xedge         xedge  0.077198024               0 0.077198024
## xbar           xbar  0.077078657               0 0.077078657
## xedgeycor xedgeycor  0.059957233               0 0.059957233
## height       height -0.027185778               0 0.027185778
## ybar           ybar -0.026531619               0 0.026531619
## width         width -0.023742242               0 0.023742242
## xbox           xbox  0.017494730               0 0.017494730
## xy2bar       xy2bar -0.013432353               0 0.013432353
## ybox           ybox -0.010644069               0 0.010644069
## .rnorm       .rnorm -0.007715895               0 0.007715895
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
## elapsed6   6.789
## elapsed7   6.993
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, 
          myfind_cor_features(feats_df=glb_feats_df, entity_df=glb_trnent_df, 
                                rsp_var=glb_rsp_var)))
```

```
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## height     0.3631670  0.137913885  0.037032756  0.00696237  0.72507621
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## width      0.3044023  0.237091223 -0.055504629 -0.07981959  0.74958603
## xbox       0.4189202  0.258062943  0.066381893  0.03539408  0.58986314
## xy2bar     0.1053847 -0.167797326 -0.408750624  0.19712757  0.06289209
## ybox       0.3301187  0.182521785 -0.049441651  0.03925331  0.59233999
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## height     0.15048782  0.03080905  0.447306032  0.070654856  0.10651721
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## width     -0.13267852  0.16583046  0.535571914  0.251275207  0.06511370
## xbox       0.11162368  0.25185573  0.561510393  0.102252624  0.05748968
## xy2bar    -0.18791695 -0.47276984  0.388005008  0.405921800 -0.72360421
## ybox       0.06926633  0.20752321  0.458347363  0.186361155 -0.02496432
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                height        ybar       width        xbox      xy2bar
## yedge      0.36316699  0.19457273  0.30440233  0.41892019  0.10538468
## y2bar      0.13791388  0.22085879  0.23709122  0.25806294 -0.16779733
## x2ybar     0.03703276  0.59737167 -0.05550463  0.06638189 -0.40875062
## yedgexcor  0.00696237  0.07203741 -0.07981959  0.03539408  0.19712757
## onpix      0.72507621  0.06046276  0.74958603  0.58986314  0.06289209
## x2bar      0.15048782  0.57592372 -0.13267852  0.11162368 -0.18791695
## xybar      0.03080905  0.25500038  0.16583046  0.25185573 -0.47276984
## xedge      0.44730603 -0.16090656  0.53557191  0.56151039  0.38800501
## xbar       0.07065486 -0.68891175  0.25127521  0.10225262  0.40592180
## xedgeycor  0.10651721  0.63447716  0.06511370  0.05748968 -0.72360421
## height     1.00000000  0.05106298  0.74581813  0.67678779  0.06391829
## ybar       0.05106298  1.00000000 -0.03809293  0.09104240 -0.65639458
## width      0.74581813 -0.03809293  1.00000000  0.80165985  0.01445667
## xbox       0.67678779  0.09104240  0.80165985  1.00000000  0.01943607
## xy2bar     0.06391829 -0.65639458  0.01445667  0.01943607  1.00000000
## ybox       0.83365004 -0.02204225  0.72801171  0.78990028  0.09786836
## .rnorm    -0.02042485  0.03044435 -0.04288041 -0.02691526 -0.01395193
##                  ybox       .rnorm
## yedge      0.33011872 -0.015180902
## y2bar      0.18252178 -0.026653803
## x2ybar    -0.04944165  0.008129806
## yedgexcor  0.03925331  0.011246263
## onpix      0.59233999 -0.022130407
## x2bar      0.06926633  0.032552469
## xybar      0.20752321 -0.011854737
## xedge      0.45834736 -0.009893245
## xbar       0.18636115 -0.023121138
## xedgeycor -0.02496432  0.017453770
## height     0.83365004 -0.020424852
## ybar      -0.02204225  0.030444350
## width      0.72801171 -0.042880406
## xbox       0.78990028 -0.026915256
## xy2bar     0.09786836 -0.013951935
## ybox       1.00000000 -0.014201592
## .rnorm    -0.01420159  1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## height    0.3631670 0.137913885 0.037032756 0.00696237 0.72507621
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## width     0.3044023 0.237091223 0.055504629 0.07981959 0.74958603
## xbox      0.4189202 0.258062943 0.066381893 0.03539408 0.58986314
## xy2bar    0.1053847 0.167797326 0.408750624 0.19712757 0.06289209
## ybox      0.3301187 0.182521785 0.049441651 0.03925331 0.59233999
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## height    0.15048782 0.03080905 0.447306032 0.070654856 0.10651721
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## width     0.13267852 0.16583046 0.535571914 0.251275207 0.06511370
## xbox      0.11162368 0.25185573 0.561510393 0.102252624 0.05748968
## xy2bar    0.18791695 0.47276984 0.388005008 0.405921800 0.72360421
## ybox      0.06926633 0.20752321 0.458347363 0.186361155 0.02496432
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##               height       ybar      width       xbox     xy2bar
## yedge     0.36316699 0.19457273 0.30440233 0.41892019 0.10538468
## y2bar     0.13791388 0.22085879 0.23709122 0.25806294 0.16779733
## x2ybar    0.03703276 0.59737167 0.05550463 0.06638189 0.40875062
## yedgexcor 0.00696237 0.07203741 0.07981959 0.03539408 0.19712757
## onpix     0.72507621 0.06046276 0.74958603 0.58986314 0.06289209
## x2bar     0.15048782 0.57592372 0.13267852 0.11162368 0.18791695
## xybar     0.03080905 0.25500038 0.16583046 0.25185573 0.47276984
## xedge     0.44730603 0.16090656 0.53557191 0.56151039 0.38800501
## xbar      0.07065486 0.68891175 0.25127521 0.10225262 0.40592180
## xedgeycor 0.10651721 0.63447716 0.06511370 0.05748968 0.72360421
## height    0.00000000 0.05106298 0.74581813 0.67678779 0.06391829
## ybar      0.05106298 0.00000000 0.03809293 0.09104240 0.65639458
## width     0.74581813 0.03809293 0.00000000 0.80165985 0.01445667
## xbox      0.67678779 0.09104240 0.80165985 0.00000000 0.01943607
## xy2bar    0.06391829 0.65639458 0.01445667 0.01943607 0.00000000
## ybox      0.83365004 0.02204225 0.72801171 0.78990028 0.09786836
## .rnorm    0.02042485 0.03044435 0.04288041 0.02691526 0.01395193
##                 ybox      .rnorm
## yedge     0.33011872 0.015180902
## y2bar     0.18252178 0.026653803
## x2ybar    0.04944165 0.008129806
## yedgexcor 0.03925331 0.011246263
## onpix     0.59233999 0.022130407
## x2bar     0.06926633 0.032552469
## xybar     0.20752321 0.011854737
## xedge     0.45834736 0.009893245
## xbar      0.18636115 0.023121138
## xedgeycor 0.02496432 0.017453770
## height    0.83365004 0.020424852
## ybar      0.02204225 0.030444350
## width     0.72801171 0.042880406
## xbox      0.78990028 0.026915256
## xy2bar    0.09786836 0.013951935
## ybox      0.00000000 0.014201592
## .rnorm    0.01420159 0.000000000
## [1] "cor(height, ybox)=0.8337"
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-1.png) 

```
## [1] "cor(isB, height)=-0.0272"
## [1] "cor(isB, ybox)=-0.0106"
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

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-2.png) 

```
## [1] "checking correlations for features:"
##  [1] "yedge"     "y2bar"     "x2ybar"    "yedgexcor" "onpix"    
##  [6] "x2bar"     "xybar"     "xedge"     "xbar"      "xedgeycor"
## [11] "height"    "ybar"      "width"     "xbox"      "xy2bar"   
## [16] ".rnorm"   
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## height     0.3631670  0.137913885  0.037032756  0.00696237  0.72507621
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## width      0.3044023  0.237091223 -0.055504629 -0.07981959  0.74958603
## xbox       0.4189202  0.258062943  0.066381893  0.03539408  0.58986314
## xy2bar     0.1053847 -0.167797326 -0.408750624  0.19712757  0.06289209
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## height     0.15048782  0.03080905  0.447306032  0.070654856  0.10651721
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## width     -0.13267852  0.16583046  0.535571914  0.251275207  0.06511370
## xbox       0.11162368  0.25185573  0.561510393  0.102252624  0.05748968
## xy2bar    -0.18791695 -0.47276984  0.388005008  0.405921800 -0.72360421
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                height        ybar       width        xbox      xy2bar
## yedge      0.36316699  0.19457273  0.30440233  0.41892019  0.10538468
## y2bar      0.13791388  0.22085879  0.23709122  0.25806294 -0.16779733
## x2ybar     0.03703276  0.59737167 -0.05550463  0.06638189 -0.40875062
## yedgexcor  0.00696237  0.07203741 -0.07981959  0.03539408  0.19712757
## onpix      0.72507621  0.06046276  0.74958603  0.58986314  0.06289209
## x2bar      0.15048782  0.57592372 -0.13267852  0.11162368 -0.18791695
## xybar      0.03080905  0.25500038  0.16583046  0.25185573 -0.47276984
## xedge      0.44730603 -0.16090656  0.53557191  0.56151039  0.38800501
## xbar       0.07065486 -0.68891175  0.25127521  0.10225262  0.40592180
## xedgeycor  0.10651721  0.63447716  0.06511370  0.05748968 -0.72360421
## height     1.00000000  0.05106298  0.74581813  0.67678779  0.06391829
## ybar       0.05106298  1.00000000 -0.03809293  0.09104240 -0.65639458
## width      0.74581813 -0.03809293  1.00000000  0.80165985  0.01445667
## xbox       0.67678779  0.09104240  0.80165985  1.00000000  0.01943607
## xy2bar     0.06391829 -0.65639458  0.01445667  0.01943607  1.00000000
## .rnorm    -0.02042485  0.03044435 -0.04288041 -0.02691526 -0.01395193
##                 .rnorm
## yedge     -0.015180902
## y2bar     -0.026653803
## x2ybar     0.008129806
## yedgexcor  0.011246263
## onpix     -0.022130407
## x2bar      0.032552469
## xybar     -0.011854737
## xedge     -0.009893245
## xbar      -0.023121138
## xedgeycor  0.017453770
## height    -0.020424852
## ybar       0.030444350
## width     -0.042880406
## xbox      -0.026915256
## xy2bar    -0.013951935
## .rnorm     1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## height    0.3631670 0.137913885 0.037032756 0.00696237 0.72507621
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## width     0.3044023 0.237091223 0.055504629 0.07981959 0.74958603
## xbox      0.4189202 0.258062943 0.066381893 0.03539408 0.58986314
## xy2bar    0.1053847 0.167797326 0.408750624 0.19712757 0.06289209
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## height    0.15048782 0.03080905 0.447306032 0.070654856 0.10651721
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## width     0.13267852 0.16583046 0.535571914 0.251275207 0.06511370
## xbox      0.11162368 0.25185573 0.561510393 0.102252624 0.05748968
## xy2bar    0.18791695 0.47276984 0.388005008 0.405921800 0.72360421
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##               height       ybar      width       xbox     xy2bar
## yedge     0.36316699 0.19457273 0.30440233 0.41892019 0.10538468
## y2bar     0.13791388 0.22085879 0.23709122 0.25806294 0.16779733
## x2ybar    0.03703276 0.59737167 0.05550463 0.06638189 0.40875062
## yedgexcor 0.00696237 0.07203741 0.07981959 0.03539408 0.19712757
## onpix     0.72507621 0.06046276 0.74958603 0.58986314 0.06289209
## x2bar     0.15048782 0.57592372 0.13267852 0.11162368 0.18791695
## xybar     0.03080905 0.25500038 0.16583046 0.25185573 0.47276984
## xedge     0.44730603 0.16090656 0.53557191 0.56151039 0.38800501
## xbar      0.07065486 0.68891175 0.25127521 0.10225262 0.40592180
## xedgeycor 0.10651721 0.63447716 0.06511370 0.05748968 0.72360421
## height    0.00000000 0.05106298 0.74581813 0.67678779 0.06391829
## ybar      0.05106298 0.00000000 0.03809293 0.09104240 0.65639458
## width     0.74581813 0.03809293 0.00000000 0.80165985 0.01445667
## xbox      0.67678779 0.09104240 0.80165985 0.00000000 0.01943607
## xy2bar    0.06391829 0.65639458 0.01445667 0.01943607 0.00000000
## .rnorm    0.02042485 0.03044435 0.04288041 0.02691526 0.01395193
##                .rnorm
## yedge     0.015180902
## y2bar     0.026653803
## x2ybar    0.008129806
## yedgexcor 0.011246263
## onpix     0.022130407
## x2bar     0.032552469
## xybar     0.011854737
## xedge     0.009893245
## xbar      0.023121138
## xedgeycor 0.017453770
## height    0.020424852
## ybar      0.030444350
## width     0.042880406
## xbox      0.026915256
## xy2bar    0.013951935
## .rnorm    0.000000000
## [1] "cor(width, xbox)=0.8017"
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-3.png) 

```
## [1] "cor(isB, width)=-0.0237"
## [1] "cor(isB, xbox)=0.0175"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified xbox as highly correlated with other features
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-4.png) 

```
## [1] "checking correlations for features:"
##  [1] "yedge"     "y2bar"     "x2ybar"    "yedgexcor" "onpix"    
##  [6] "x2bar"     "xybar"     "xedge"     "xbar"      "xedgeycor"
## [11] "height"    "ybar"      "width"     "xy2bar"    ".rnorm"   
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## height     0.3631670  0.137913885  0.037032756  0.00696237  0.72507621
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## width      0.3044023  0.237091223 -0.055504629 -0.07981959  0.74958603
## xy2bar     0.1053847 -0.167797326 -0.408750624  0.19712757  0.06289209
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## height     0.15048782  0.03080905  0.447306032  0.070654856  0.10651721
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## width     -0.13267852  0.16583046  0.535571914  0.251275207  0.06511370
## xy2bar    -0.18791695 -0.47276984  0.388005008  0.405921800 -0.72360421
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                height        ybar       width      xy2bar       .rnorm
## yedge      0.36316699  0.19457273  0.30440233  0.10538468 -0.015180902
## y2bar      0.13791388  0.22085879  0.23709122 -0.16779733 -0.026653803
## x2ybar     0.03703276  0.59737167 -0.05550463 -0.40875062  0.008129806
## yedgexcor  0.00696237  0.07203741 -0.07981959  0.19712757  0.011246263
## onpix      0.72507621  0.06046276  0.74958603  0.06289209 -0.022130407
## x2bar      0.15048782  0.57592372 -0.13267852 -0.18791695  0.032552469
## xybar      0.03080905  0.25500038  0.16583046 -0.47276984 -0.011854737
## xedge      0.44730603 -0.16090656  0.53557191  0.38800501 -0.009893245
## xbar       0.07065486 -0.68891175  0.25127521  0.40592180 -0.023121138
## xedgeycor  0.10651721  0.63447716  0.06511370 -0.72360421  0.017453770
## height     1.00000000  0.05106298  0.74581813  0.06391829 -0.020424852
## ybar       0.05106298  1.00000000 -0.03809293 -0.65639458  0.030444350
## width      0.74581813 -0.03809293  1.00000000  0.01445667 -0.042880406
## xy2bar     0.06391829 -0.65639458  0.01445667  1.00000000 -0.013951935
## .rnorm    -0.02042485  0.03044435 -0.04288041 -0.01395193  1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## height    0.3631670 0.137913885 0.037032756 0.00696237 0.72507621
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## width     0.3044023 0.237091223 0.055504629 0.07981959 0.74958603
## xy2bar    0.1053847 0.167797326 0.408750624 0.19712757 0.06289209
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## height    0.15048782 0.03080905 0.447306032 0.070654856 0.10651721
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## width     0.13267852 0.16583046 0.535571914 0.251275207 0.06511370
## xy2bar    0.18791695 0.47276984 0.388005008 0.405921800 0.72360421
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##               height       ybar      width     xy2bar      .rnorm
## yedge     0.36316699 0.19457273 0.30440233 0.10538468 0.015180902
## y2bar     0.13791388 0.22085879 0.23709122 0.16779733 0.026653803
## x2ybar    0.03703276 0.59737167 0.05550463 0.40875062 0.008129806
## yedgexcor 0.00696237 0.07203741 0.07981959 0.19712757 0.011246263
## onpix     0.72507621 0.06046276 0.74958603 0.06289209 0.022130407
## x2bar     0.15048782 0.57592372 0.13267852 0.18791695 0.032552469
## xybar     0.03080905 0.25500038 0.16583046 0.47276984 0.011854737
## xedge     0.44730603 0.16090656 0.53557191 0.38800501 0.009893245
## xbar      0.07065486 0.68891175 0.25127521 0.40592180 0.023121138
## xedgeycor 0.10651721 0.63447716 0.06511370 0.72360421 0.017453770
## height    0.00000000 0.05106298 0.74581813 0.06391829 0.020424852
## ybar      0.05106298 0.00000000 0.03809293 0.65639458 0.030444350
## width     0.74581813 0.03809293 0.00000000 0.01445667 0.042880406
## xy2bar    0.06391829 0.65639458 0.01445667 0.00000000 0.013951935
## .rnorm    0.02042485 0.03044435 0.04288041 0.01395193 0.000000000
## [1] "cor(onpix, width)=0.7496"
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-5.png) 

```
## [1] "cor(isB, onpix)=0.2161"
## [1] "cor(isB, width)=-0.0237"
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

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-6.png) 

```
## [1] "checking correlations for features:"
##  [1] "yedge"     "y2bar"     "x2ybar"    "yedgexcor" "onpix"    
##  [6] "x2bar"     "xybar"     "xedge"     "xbar"      "xedgeycor"
## [11] "height"    "ybar"      "xy2bar"    ".rnorm"   
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## height     0.3631670  0.137913885  0.037032756  0.00696237  0.72507621
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## xy2bar     0.1053847 -0.167797326 -0.408750624  0.19712757  0.06289209
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## height     0.15048782  0.03080905  0.447306032  0.070654856  0.10651721
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## xy2bar    -0.18791695 -0.47276984  0.388005008  0.405921800 -0.72360421
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                height        ybar      xy2bar       .rnorm
## yedge      0.36316699  0.19457273  0.10538468 -0.015180902
## y2bar      0.13791388  0.22085879 -0.16779733 -0.026653803
## x2ybar     0.03703276  0.59737167 -0.40875062  0.008129806
## yedgexcor  0.00696237  0.07203741  0.19712757  0.011246263
## onpix      0.72507621  0.06046276  0.06289209 -0.022130407
## x2bar      0.15048782  0.57592372 -0.18791695  0.032552469
## xybar      0.03080905  0.25500038 -0.47276984 -0.011854737
## xedge      0.44730603 -0.16090656  0.38800501 -0.009893245
## xbar       0.07065486 -0.68891175  0.40592180 -0.023121138
## xedgeycor  0.10651721  0.63447716 -0.72360421  0.017453770
## height     1.00000000  0.05106298  0.06391829 -0.020424852
## ybar       0.05106298  1.00000000 -0.65639458  0.030444350
## xy2bar     0.06391829 -0.65639458  1.00000000 -0.013951935
## .rnorm    -0.02042485  0.03044435 -0.01395193  1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## height    0.3631670 0.137913885 0.037032756 0.00696237 0.72507621
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## xy2bar    0.1053847 0.167797326 0.408750624 0.19712757 0.06289209
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## height    0.15048782 0.03080905 0.447306032 0.070654856 0.10651721
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## xy2bar    0.18791695 0.47276984 0.388005008 0.405921800 0.72360421
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##               height       ybar     xy2bar      .rnorm
## yedge     0.36316699 0.19457273 0.10538468 0.015180902
## y2bar     0.13791388 0.22085879 0.16779733 0.026653803
## x2ybar    0.03703276 0.59737167 0.40875062 0.008129806
## yedgexcor 0.00696237 0.07203741 0.19712757 0.011246263
## onpix     0.72507621 0.06046276 0.06289209 0.022130407
## x2bar     0.15048782 0.57592372 0.18791695 0.032552469
## xybar     0.03080905 0.25500038 0.47276984 0.011854737
## xedge     0.44730603 0.16090656 0.38800501 0.009893245
## xbar      0.07065486 0.68891175 0.40592180 0.023121138
## xedgeycor 0.10651721 0.63447716 0.72360421 0.017453770
## height    0.00000000 0.05106298 0.06391829 0.020424852
## ybar      0.05106298 0.00000000 0.65639458 0.030444350
## xy2bar    0.06391829 0.65639458 0.00000000 0.013951935
## .rnorm    0.02042485 0.03044435 0.01395193 0.000000000
## [1] "cor(onpix, height)=0.7251"
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-7.png) 

```
## [1] "cor(isB, onpix)=0.2161"
## [1] "cor(isB, height)=-0.0272"
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

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-8.png) 

```
## [1] "checking correlations for features:"
##  [1] "yedge"     "y2bar"     "x2ybar"    "yedgexcor" "onpix"    
##  [6] "x2bar"     "xybar"     "xedge"     "xbar"      "xedgeycor"
## [11] "ybar"      "xy2bar"    ".rnorm"   
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## xy2bar     0.1053847 -0.167797326 -0.408750624  0.19712757  0.06289209
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## xy2bar    -0.18791695 -0.47276984  0.388005008  0.405921800 -0.72360421
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                  ybar      xy2bar       .rnorm
## yedge      0.19457273  0.10538468 -0.015180902
## y2bar      0.22085879 -0.16779733 -0.026653803
## x2ybar     0.59737167 -0.40875062  0.008129806
## yedgexcor  0.07203741  0.19712757  0.011246263
## onpix      0.06046276  0.06289209 -0.022130407
## x2bar      0.57592372 -0.18791695  0.032552469
## xybar      0.25500038 -0.47276984 -0.011854737
## xedge     -0.16090656  0.38800501 -0.009893245
## xbar      -0.68891175  0.40592180 -0.023121138
## xedgeycor  0.63447716 -0.72360421  0.017453770
## ybar       1.00000000 -0.65639458  0.030444350
## xy2bar    -0.65639458  1.00000000 -0.013951935
## .rnorm     0.03044435 -0.01395193  1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## xy2bar    0.1053847 0.167797326 0.408750624 0.19712757 0.06289209
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## xy2bar    0.18791695 0.47276984 0.388005008 0.405921800 0.72360421
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##                 ybar     xy2bar      .rnorm
## yedge     0.19457273 0.10538468 0.015180902
## y2bar     0.22085879 0.16779733 0.026653803
## x2ybar    0.59737167 0.40875062 0.008129806
## yedgexcor 0.07203741 0.19712757 0.011246263
## onpix     0.06046276 0.06289209 0.022130407
## x2bar     0.57592372 0.18791695 0.032552469
## xybar     0.25500038 0.47276984 0.011854737
## xedge     0.16090656 0.38800501 0.009893245
## xbar      0.68891175 0.40592180 0.023121138
## xedgeycor 0.63447716 0.72360421 0.017453770
## ybar      0.00000000 0.65639458 0.030444350
## xy2bar    0.65639458 0.00000000 0.013951935
## .rnorm    0.03044435 0.01395193 0.000000000
## [1] "cor(xedgeycor, xy2bar)=-0.7236"
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-9.png) 

```
## [1] "cor(isB, xedgeycor)=0.0600"
## [1] "cor(isB, xy2bar)=-0.0134"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified xy2bar as highly correlated with other
## features
```

![](Letter_Recognition_B_files/figure-html/remove_correlated_features-10.png) 

```
## [1] "checking correlations for features:"
##  [1] "yedge"     "y2bar"     "x2ybar"    "yedgexcor" "onpix"    
##  [6] "x2bar"     "xybar"     "xedge"     "xbar"      "xedgeycor"
## [11] "ybar"      ".rnorm"   
##                yedge        y2bar       x2ybar   yedgexcor       onpix
## yedge      1.0000000  0.563770832  0.462012418  0.18047370  0.57996512
## y2bar      0.5637708  1.000000000  0.388680283  0.31685825  0.31631656
## x2ybar     0.4620124  0.388680283  1.000000000 -0.06842878  0.30658313
## yedgexcor  0.1804737  0.316858253 -0.068428781  1.00000000 -0.06703528
## onpix      0.5799651  0.316316557  0.306583134 -0.06703528  1.00000000
## x2bar      0.4947506  0.213638569  0.423090534  0.31900374  0.03570184
## xybar     -0.2698520  0.049563972  0.056790618  0.06118085 -0.11057574
## xedge      0.4667461  0.110210193  0.082498376 -0.08178763  0.65417545
## xbar      -0.1313813  0.002694833 -0.474503188  0.08603556  0.12806919
## xedgeycor  0.1143824  0.109627914  0.593154379 -0.24272026  0.17929877
## ybar       0.1945727  0.220858790  0.597371675  0.07203741  0.06046276
## .rnorm    -0.0151809 -0.026653803  0.008129806  0.01124626 -0.02213041
##                 x2bar       xybar        xedge         xbar   xedgeycor
## yedge      0.49475057 -0.26985197  0.466746104 -0.131381325  0.11438241
## y2bar      0.21363857  0.04956397  0.110210193  0.002694833  0.10962791
## x2ybar     0.42309053  0.05679062  0.082498376 -0.474503188  0.59315438
## yedgexcor  0.31900374  0.06118085 -0.081787630  0.086035558 -0.24272026
## onpix      0.03570184 -0.11057574  0.654175447  0.128069194  0.17929877
## x2bar      1.00000000 -0.13374007 -0.126699366 -0.564631935  0.28834505
## xybar     -0.13374007  1.00000000 -0.195646132  0.197798744  0.29288927
## xedge     -0.12669937 -0.19564613  1.000000000  0.225139904 -0.11775813
## xbar      -0.56463194  0.19779874  0.225139904  1.000000000 -0.44403521
## xedgeycor  0.28834505  0.29288927 -0.117758133 -0.444035205  1.00000000
## ybar       0.57592372  0.25500038 -0.160906556 -0.688911750  0.63447716
## .rnorm     0.03255247 -0.01185474 -0.009893245 -0.023121138  0.01745377
##                  ybar       .rnorm
## yedge      0.19457273 -0.015180902
## y2bar      0.22085879 -0.026653803
## x2ybar     0.59737167  0.008129806
## yedgexcor  0.07203741  0.011246263
## onpix      0.06046276 -0.022130407
## x2bar      0.57592372  0.032552469
## xybar      0.25500038 -0.011854737
## xedge     -0.16090656 -0.009893245
## xbar      -0.68891175 -0.023121138
## xedgeycor  0.63447716  0.017453770
## ybar       1.00000000  0.030444350
## .rnorm     0.03044435  1.000000000
##               yedge       y2bar      x2ybar  yedgexcor      onpix
## yedge     0.0000000 0.563770832 0.462012418 0.18047370 0.57996512
## y2bar     0.5637708 0.000000000 0.388680283 0.31685825 0.31631656
## x2ybar    0.4620124 0.388680283 0.000000000 0.06842878 0.30658313
## yedgexcor 0.1804737 0.316858253 0.068428781 0.00000000 0.06703528
## onpix     0.5799651 0.316316557 0.306583134 0.06703528 0.00000000
## x2bar     0.4947506 0.213638569 0.423090534 0.31900374 0.03570184
## xybar     0.2698520 0.049563972 0.056790618 0.06118085 0.11057574
## xedge     0.4667461 0.110210193 0.082498376 0.08178763 0.65417545
## xbar      0.1313813 0.002694833 0.474503188 0.08603556 0.12806919
## xedgeycor 0.1143824 0.109627914 0.593154379 0.24272026 0.17929877
## ybar      0.1945727 0.220858790 0.597371675 0.07203741 0.06046276
## .rnorm    0.0151809 0.026653803 0.008129806 0.01124626 0.02213041
##                x2bar      xybar       xedge        xbar  xedgeycor
## yedge     0.49475057 0.26985197 0.466746104 0.131381325 0.11438241
## y2bar     0.21363857 0.04956397 0.110210193 0.002694833 0.10962791
## x2ybar    0.42309053 0.05679062 0.082498376 0.474503188 0.59315438
## yedgexcor 0.31900374 0.06118085 0.081787630 0.086035558 0.24272026
## onpix     0.03570184 0.11057574 0.654175447 0.128069194 0.17929877
## x2bar     0.00000000 0.13374007 0.126699366 0.564631935 0.28834505
## xybar     0.13374007 0.00000000 0.195646132 0.197798744 0.29288927
## xedge     0.12669937 0.19564613 0.000000000 0.225139904 0.11775813
## xbar      0.56463194 0.19779874 0.225139904 0.000000000 0.44403521
## xedgeycor 0.28834505 0.29288927 0.117758133 0.444035205 0.00000000
## ybar      0.57592372 0.25500038 0.160906556 0.688911750 0.63447716
## .rnorm    0.03255247 0.01185474 0.009893245 0.023121138 0.01745377
##                 ybar      .rnorm
## yedge     0.19457273 0.015180902
## y2bar     0.22085879 0.026653803
## x2ybar    0.59737167 0.008129806
## yedgexcor 0.07203741 0.011246263
## onpix     0.06046276 0.022130407
## x2bar     0.57592372 0.032552469
## xybar     0.25500038 0.011854737
## xedge     0.16090656 0.009893245
## xbar      0.68891175 0.023121138
## xedgeycor 0.63447716 0.017453770
## ybar      0.00000000 0.030444350
## .rnorm    0.03044435 0.000000000
##                  id        cor.y exclude.as.feat   cor.y.abs cor.low
## yedge         yedge  0.561065858               0 0.561065858       1
## y2bar         y2bar  0.517624377               0 0.517624377       1
## x2ybar       x2ybar  0.286395882               0 0.286395882       1
## yedgexcor yedgexcor  0.219787666               0 0.219787666       1
## onpix         onpix  0.216051154               0 0.216051154       1
## x2bar         x2bar  0.147216267               0 0.147216267       1
## xedge         xedge  0.077198024               0 0.077198024       1
## xbar           xbar  0.077078657               0 0.077078657       1
## xedgeycor xedgeycor  0.059957233               0 0.059957233       1
## xbox           xbox  0.017494730               0 0.017494730       0
## .rnorm       .rnorm -0.007715895               0 0.007715895       1
## ybox           ybox -0.010644069               0 0.010644069       0
## xy2bar       xy2bar -0.013432353               0 0.013432353       0
## width         width -0.023742242               0 0.023742242       0
## ybar           ybar -0.026531619               0 0.026531619       1
## height       height -0.027185778               0 0.027185778       0
## xybar         xybar -0.146207921               0 0.146207921       1
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
## elapsed7   6.993
## elapsed8  11.228
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
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##        N        Y 
## 0.754172 0.245828 
## [1] "MFO.val:"
## [1] "N"
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
```

```
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## 
## The following object is masked from 'package:stats':
## 
##     lowess
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##          N        Y
## 1 0.754172 0.245828
## 2 0.754172 0.245828
## 3 0.754172 0.245828
## 4 0.754172 0.245828
## 5 0.754172 0.245828
## 6 0.754172 0.245828
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   isB isB.predict.MFO.myMFO_classfr.N
## 1   N                            1175
## 2   Y                             383
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.MFO.myMFO_classfr.N isB.predict.MFO.myMFO_classfr.Y
## 1   N                            1175                               0
## 2   Y                             383                               0
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y  383    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.541720e-01   0.000000e+00   7.320038e-01   7.753782e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   5.137209e-01   7.527969e-85 
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##          N        Y
## 1 0.754172 0.245828
## 2 0.754172 0.245828
## 3 0.754172 0.245828
## 4 0.754172 0.245828
## 5 0.754172 0.245828
## 6 0.754172 0.245828
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   isB isB.predict.MFO.myMFO_classfr.N
## 1   N                            1175
## 2   Y                             383
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.MFO.myMFO_classfr.N isB.predict.MFO.myMFO_classfr.Y
## 1   N                            1175                               0
## 2   Y                             383                               0
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y  383    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.541720e-01   0.000000e+00   7.320038e-01   7.753782e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   5.137209e-01   7.527969e-85 
##            model_id  model_method  feats max.nTuningRuns
## 1 MFO.myMFO_classfr myMFO_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.346                 0.003         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0         0.754172
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7320038             0.7753782             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0         0.754172
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7320038             0.7753782             0
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
## unique.vals 2      factor     numeric  
## unique.prob 2      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
## [1] "in Random.Classifier$prob"
```

![](Letter_Recognition_B_files/figure-html/fit.models-1.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction   N   Y
##          N 882 289
##          Y 293  94
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   882
## 2   Y                                   289
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   293
## 2                                    94
##           Reference
## Prediction   N   Y
##          N 882 289
##          Y 293  94
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   882
## 2   Y                                   289
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   293
## 2                                    94
##           Reference
## Prediction   N   Y
##          N 882 289
##          Y 293  94
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   882
## 2   Y                                   289
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   293
## 2                                    94
##           Reference
## Prediction   N   Y
##          N 882 289
##          Y 293  94
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   882
## 2   Y                                   289
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   293
## 2                                    94
##           Reference
## Prediction   N   Y
##          N 882 289
##          Y 293  94
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   882
## 2   Y                                   289
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   293
## 2                                    94
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.3946419
## 3        0.2 0.3946419
## 4        0.3 0.2441558
## 5        0.4 0.2441558
## 6        0.5 0.2441558
## 7        0.6 0.2441558
## 8        0.7 0.2441558
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-2.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   isB isB.predict.Random.myrandom_classfr.Y
## 1   N                                  1175
## 2   Y                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##          Prediction
## Reference    N    Y
##         N    0 1175
##         Y    0  383
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   2.458280e-01   0.000000e+00   2.246218e-01   2.679962e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  4.498037e-257 
## [1] "in Random.Classifier$prob"
```

![](Letter_Recognition_B_files/figure-html/fit.models-3.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction   N   Y
##          N 893 294
##          Y 282  89
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   893
## 2   Y                                   294
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   282
## 2                                    89
##           Reference
## Prediction   N   Y
##          N 893 294
##          Y 282  89
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   893
## 2   Y                                   294
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   282
## 2                                    89
##           Reference
## Prediction   N   Y
##          N 893 294
##          Y 282  89
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   893
## 2   Y                                   294
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   282
## 2                                    89
##           Reference
## Prediction   N   Y
##          N 893 294
##          Y 282  89
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   893
## 2   Y                                   294
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   282
## 2                                    89
##           Reference
## Prediction   N   Y
##          N 893 294
##          Y 282  89
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                   893
## 2   Y                                   294
##   isB.predict.Random.myrandom_classfr.Y
## 1                                   282
## 2                                    89
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Random.myrandom_classfr.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.3946419
## 3        0.2 0.3946419
## 4        0.3 0.2360743
## 5        0.4 0.2360743
## 6        0.5 0.2360743
## 7        0.6 0.2360743
## 8        0.7 0.2360743
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-4.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   isB isB.predict.Random.myrandom_classfr.Y
## 1   N                                  1175
## 2   Y                                   383
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Random.myrandom_classfr.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Random.myrandom_classfr.Y
## 1                                  1175
## 2                                   383
##          Prediction
## Reference    N    Y
##         N    0 1175
##         Y    0  383
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   2.458280e-01   0.000000e+00   2.246218e-01   2.679962e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  4.498037e-257 
##                  model_id     model_method  feats max.nTuningRuns
## 1 Random.myrandom_classfr myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.221                 0.001   0.4980346
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.3946419         0.245828
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.2246218             0.2679962             0    0.496188
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.3946419         0.245828
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.2246218             0.2679962             0
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
## [1] "    indep_vars: yedge"
```

```
## Loading required package: rpart
```

```
## Fitting cp = 0.111 on full training set
```

```
## Loading required package: rpart.plot
```

![](Letter_Recognition_B_files/figure-html/fit.models-5.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##          CP nsplit rel error
## 1 0.1109661      0         1
## 
## Node number 1: 1558 observations
##   predicted class=N  expected loss=0.245828  P(node) =1
##     class counts:  1175   383
##    probabilities: 0.754 0.246 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 383 N (0.7541720 0.2458280) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   isB isB.predict.Max.cor.Y.cv.0.rpart.N
## 1   N                               1175
## 2   Y                                383
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.rpart.N
## 1   N                               1175
## 2   Y                                383
##   isB.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                  0
## 2                                  0
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y  383    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.541720e-01   0.000000e+00   7.320038e-01   7.753782e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   5.137209e-01   7.527969e-85 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   isB isB.predict.Max.cor.Y.cv.0.rpart.N
## 1   N                               1175
## 2   Y                                383
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.rpart.N
## 1   N                               1175
## 2   Y                                383
##   isB.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                  0
## 2                                  0
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y  383    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.541720e-01   0.000000e+00   7.320038e-01   7.753782e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   5.137209e-01   7.527969e-85 
##               model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.rpart        rpart yedge               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                        0.8                 0.024         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0         0.754172
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7320038             0.7753782             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0         0.754172
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7320038             0.7753782             0
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
## [1] "    indep_vars: yedge"
## Fitting cp = 0 on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-6.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##            CP nsplit rel error
## 1 0.110966057      0 1.0000000
## 2 0.023498695      2 0.7780679
## 3 0.002610966      3 0.7545692
## 4 0.000000000      4 0.7519582
## 
## Variable importance
## yedge 
##   100 
## 
## Node number 1: 1558 observations,    complexity param=0.1109661
##   predicted class=N  expected loss=0.245828  P(node) =1
##     class counts:  1175   383
##    probabilities: 0.754 0.246 
##   left son=2 (824 obs) right son=3 (734 obs)
##   Primary splits:
##       yedge < 4.5  to the left,  improve=151.6414, (0 missing)
## 
## Node number 2: 824 observations
##   predicted class=N  expected loss=0.03762136  P(node) =0.5288832
##     class counts:   793    31
##    probabilities: 0.962 0.038 
## 
## Node number 3: 734 observations,    complexity param=0.1109661
##   predicted class=N  expected loss=0.479564  P(node) =0.4711168
##     class counts:   382   352
##    probabilities: 0.520 0.480 
##   left son=6 (605 obs) right son=7 (129 obs)
##   Primary splits:
##       yedge < 7.5  to the left,  improve=38.32055, (0 missing)
## 
## Node number 6: 605 observations,    complexity param=0.02349869
##   predicted class=N  expected loss=0.4049587  P(node) =0.3883184
##     class counts:   360   245
##    probabilities: 0.595 0.405 
##   left son=12 (466 obs) right son=13 (139 obs)
##   Primary splits:
##       yedge < 6.5  to the left,  improve=5.859469, (0 missing)
## 
## Node number 7: 129 observations,    complexity param=0.002610966
##   predicted class=Y  expected loss=0.1705426  P(node) =0.08279846
##     class counts:    22   107
##    probabilities: 0.171 0.829 
##   left son=14 (7 obs) right son=15 (122 obs)
##   Primary splits:
##       yedge < 10.5 to the right, improve=2.379028, (0 missing)
## 
## Node number 12: 466 observations
##   predicted class=N  expected loss=0.3669528  P(node) =0.2991014
##     class counts:   295   171
##    probabilities: 0.633 0.367 
## 
## Node number 13: 139 observations
##   predicted class=Y  expected loss=0.4676259  P(node) =0.08921694
##     class counts:    65    74
##    probabilities: 0.468 0.532 
## 
## Node number 14: 7 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.00449294
##     class counts:     4     3
##    probabilities: 0.571 0.429 
## 
## Node number 15: 122 observations
##   predicted class=Y  expected loss=0.147541  P(node) =0.07830552
##     class counts:    18   104
##    probabilities: 0.148 0.852 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1558 383 N (0.75417202 0.24582798)  
##    2) yedge< 4.5 824  31 N (0.96237864 0.03762136) *
##    3) yedge>=4.5 734 352 N (0.52043597 0.47956403)  
##      6) yedge< 7.5 605 245 N (0.59504132 0.40495868)  
##       12) yedge< 6.5 466 171 N (0.63304721 0.36695279) *
##       13) yedge>=6.5 139  65 Y (0.46762590 0.53237410) *
##      7) yedge>=7.5 129  22 Y (0.17054264 0.82945736)  
##       14) yedge>=10.5 7   3 N (0.57142857 0.42857143) *
##       15) yedge< 10.5 122  18 Y (0.14754098 0.85245902) *
```

![](Letter_Recognition_B_files/figure-html/fit.models-7.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                       0
## 2   Y                                       0
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                    1175
## 2                                     383
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     793
## 2   Y                                      31
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     382
## 2                                     352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     793
## 2   Y                                      31
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     382
## 2                                     352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     793
## 2   Y                                      31
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     382
## 2                                     352
##           Reference
## Prediction    N    Y
##          N 1088  202
##          Y   87  181
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1088
## 2   Y                                     202
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      87
## 2                                     181
##           Reference
## Prediction    N    Y
##          N 1092  205
##          Y   83  178
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1092
## 2   Y                                     205
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      83
## 2                                     178
##           Reference
## Prediction    N    Y
##          N 1157  279
##          Y   18  104
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1157
## 2   Y                                     279
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      18
## 2                                     104
##           Reference
## Prediction    N    Y
##          N 1157  279
##          Y   18  104
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1157
## 2   Y                                     279
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      18
## 2                                     104
##           Reference
## Prediction    N    Y
##          N 1157  279
##          Y   18  104
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1157
## 2   Y                                     279
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      18
## 2                                     104
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1175
## 2   Y                                     383
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                       0
## 2                                       0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1175
## 2   Y                                     383
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                       0
## 2                                       0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6302596
## 3        0.2 0.6302596
## 4        0.3 0.6302596
## 5        0.4 0.5560676
## 6        0.5 0.5527950
## 7        0.6 0.4118812
## 8        0.7 0.4118812
## 9        0.8 0.4118812
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-8.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.fit"
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     793
## 2   Y                                      31
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     382
## 2                                     352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     793
## 2   Y                                      31
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     382
## 2                                     352
##          Prediction
## Reference   N   Y
##         N 793 382
##         Y  31 352
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.349166e-01   4.537937e-01   7.122496e-01   7.566947e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   9.627904e-01   1.804326e-66
```

![](Letter_Recognition_B_files/figure-html/fit.models-9.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                       0
## 2   Y                                       0
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                    1175
## 2                                     383
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     810
## 2   Y                                      36
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     365
## 2                                     347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     810
## 2   Y                                      36
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     365
## 2                                     347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     810
## 2   Y                                      36
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     365
## 2                                     347
##           Reference
## Prediction    N    Y
##          N 1076  185
##          Y   99  198
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1076
## 2   Y                                     185
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      99
## 2                                     198
##           Reference
## Prediction    N    Y
##          N 1076  186
##          Y   99  197
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1076
## 2   Y                                     186
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      99
## 2                                     197
##           Reference
## Prediction    N    Y
##          N 1147  277
##          Y   28  106
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1147
## 2   Y                                     277
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      28
## 2                                     106
##           Reference
## Prediction    N    Y
##          N 1147  277
##          Y   28  106
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1147
## 2   Y                                     277
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      28
## 2                                     106
##           Reference
## Prediction    N    Y
##          N 1147  277
##          Y   28  106
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1147
## 2   Y                                     277
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                      28
## 2                                     106
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1175
## 2   Y                                     383
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                       0
## 2                                       0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                    1175
## 2   Y                                     383
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                       0
## 2                                       0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6337900
## 3        0.2 0.6337900
## 4        0.3 0.6337900
## 5        0.4 0.5823529
## 6        0.5 0.5802651
## 7        0.6 0.4100580
## 8        0.7 0.4100580
## 9        0.8 0.4100580
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-10.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     810
## 2   Y                                      36
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     365
## 2                                     347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1   N                                     810
## 2   Y                                      36
##   isB.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                     365
## 2                                     347
##          Prediction
## Reference   N   Y
##         N 810 365
##         Y  36 347
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.426187e-01   4.617023e-01   7.201446e-01   7.641747e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   8.616282e-01   2.677677e-60 
##                    model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.cp.0.rpart        rpart yedge               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.672                 0.022   0.8463174
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.3       0.6302596        0.7349166
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7122496             0.7566947     0.4537937   0.8452319
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3         0.63379        0.7426187
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7201446             0.7641747     0.4617023
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.rpart"
## [1] "    indep_vars: yedge"
## + Fold1: cp=0.002611 
## - Fold1: cp=0.002611 
## + Fold2: cp=0.002611 
## - Fold2: cp=0.002611 
## + Fold3: cp=0.002611 
## - Fold3: cp=0.002611 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0235 on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-11.png) ![](Letter_Recognition_B_files/figure-html/fit.models-12.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##           CP nsplit rel error
## 1 0.11096606      0 1.0000000
## 2 0.02349869      2 0.7780679
## 
## Variable importance
## yedge 
##   100 
## 
## Node number 1: 1558 observations,    complexity param=0.1109661
##   predicted class=N  expected loss=0.245828  P(node) =1
##     class counts:  1175   383
##    probabilities: 0.754 0.246 
##   left son=2 (824 obs) right son=3 (734 obs)
##   Primary splits:
##       yedge < 4.5 to the left,  improve=151.6414, (0 missing)
## 
## Node number 2: 824 observations
##   predicted class=N  expected loss=0.03762136  P(node) =0.5288832
##     class counts:   793    31
##    probabilities: 0.962 0.038 
## 
## Node number 3: 734 observations,    complexity param=0.1109661
##   predicted class=N  expected loss=0.479564  P(node) =0.4711168
##     class counts:   382   352
##    probabilities: 0.520 0.480 
##   left son=6 (605 obs) right son=7 (129 obs)
##   Primary splits:
##       yedge < 7.5 to the left,  improve=38.32055, (0 missing)
## 
## Node number 6: 605 observations
##   predicted class=N  expected loss=0.4049587  P(node) =0.3883184
##     class counts:   360   245
##    probabilities: 0.595 0.405 
## 
## Node number 7: 129 observations
##   predicted class=Y  expected loss=0.1705426  P(node) =0.08279846
##     class counts:    22   107
##    probabilities: 0.171 0.829 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1558 383 N (0.75417202 0.24582798)  
##   2) yedge< 4.5 824  31 N (0.96237864 0.03762136) *
##   3) yedge>=4.5 734 352 N (0.52043597 0.47956403)  
##     6) yedge< 7.5 605 245 N (0.59504132 0.40495868) *
##     7) yedge>=7.5 129  22 Y (0.17054264 0.82945736) *
```

![](Letter_Recognition_B_files/figure-html/fit.models-13.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                             0                          1175
## 2   Y                             0                           383
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1153                            22
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1153                            22
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1153                            22
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1153                            22
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1175                             0
## 2   Y                           383                             0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1175                             0
## 2   Y                           383                             0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6302596
## 3        0.2 0.6302596
## 4        0.3 0.6302596
## 5        0.4 0.6302596
## 6        0.5 0.4179688
## 7        0.6 0.4179688
## 8        0.7 0.4179688
## 9        0.8 0.4179688
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-14.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           793                           382
## 2   Y                            31                           352
##          Prediction
## Reference   N   Y
##         N 793 382
##         Y  31 352
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.349166e-01   4.537937e-01   7.122496e-01   7.566947e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   9.627904e-01   1.804326e-66
```

![](Letter_Recognition_B_files/figure-html/fit.models-15.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                             0                          1175
## 2   Y                             0                           383
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1147                            28
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1147                            28
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1147                            28
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1147                            28
## 2   Y                           276                           107
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1175                             0
## 2   Y                           383                             0
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                          1175                             0
## 2   Y                           383                             0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6337900
## 3        0.2 0.6337900
## 4        0.3 0.6337900
## 5        0.4 0.6337900
## 6        0.5 0.4131274
## 7        0.6 0.4131274
## 8        0.7 0.4131274
## 9        0.8 0.4131274
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-16.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.rpart.N isB.predict.Max.cor.Y.rpart.Y
## 1   N                           810                           365
## 2   Y                            36                           347
##          Prediction
## Reference   N   Y
##         N 810 365
##         Y  36 347
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.426187e-01   4.617023e-01   7.201446e-01   7.641747e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   8.616282e-01   2.677677e-60 
##          model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.rpart        rpart yedge               3
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.188                 0.023   0.8337859
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.4       0.6302596        0.8074391
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7122496             0.7566947     0.3893568   0.8302805
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4         0.63379        0.7426187
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7201446             0.7641747     0.4617023
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01217455       0.1061354
```

```r
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
## [1] "fitting model: Max.cor.Y.glm"
## [1] "    indep_vars: yedge"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-17.png) ![](Letter_Recognition_B_files/figure-html/fit.models-18.png) ![](Letter_Recognition_B_files/figure-html/fit.models-19.png) ![](Letter_Recognition_B_files/figure-html/fit.models-20.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.7887  -0.4598  -0.3028  -0.1286   2.8090  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -5.65766    0.28262  -20.02   <2e-16 ***
## yedge        0.86593    0.04869   17.78   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1737.8  on 1557  degrees of freedom
## Residual deviance: 1193.5  on 1556  degrees of freedom
## AIC: 1197.5
## 
## Number of Fisher Scoring iterations: 5
```

![](Letter_Recognition_B_files/figure-html/fit.models-21.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                           0                        1175
## 2   Y                           0                         383
##           Reference
## Prediction   N   Y
##          N 492   7
##          Y 683 376
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         492                         683
## 2   Y                           7                         376
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         793                         382
## 2   Y                          31                         352
##           Reference
## Prediction   N   Y
##          N 942 101
##          Y 233 282
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         942                         233
## 2   Y                         101                         282
##           Reference
## Prediction    N    Y
##          N 1088  202
##          Y   87  181
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1088                          87
## 2   Y                         202                         181
##           Reference
## Prediction    N    Y
##          N 1088  202
##          Y   87  181
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1088                          87
## 2   Y                         202                         181
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1153                          22
## 2   Y                         276                         107
##           Reference
## Prediction    N    Y
##          N 1153  276
##          Y   22  107
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1153                          22
## 2   Y                         276                         107
##           Reference
## Prediction    N    Y
##          N 1169  319
##          Y    6   64
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1169                           6
## 2   Y                         319                          64
##           Reference
## Prediction    N    Y
##          N 1170  372
##          Y    5   11
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1170                           5
## 2   Y                         372                          11
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1175                           0
## 2   Y                         383                           0
##    threshold    f.score
## 1        0.0 0.39464194
## 2        0.1 0.52149792
## 3        0.2 0.63025962
## 4        0.3 0.62806236
## 5        0.4 0.55606759
## 6        0.5 0.55606759
## 7        0.6 0.41796875
## 8        0.7 0.41796875
## 9        0.8 0.28256071
## 10       0.9 0.05513784
## 11       1.0 0.00000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-22.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         793                         382
## 2   Y                          31                         352
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         793                         382
## 2   Y                          31                         352
##          Prediction
## Reference   N   Y
##         N 793 382
##         Y  31 352
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.349166e-01   4.537937e-01   7.122496e-01   7.566947e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   9.627904e-01   1.804326e-66
```

![](Letter_Recognition_B_files/figure-html/fit.models-23.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                           0                        1175
## 2   Y                           0                         383
##           Reference
## Prediction   N   Y
##          N 526   4
##          Y 649 379
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         526                         649
## 2   Y                           4                         379
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         810                         365
## 2   Y                          36                         347
##           Reference
## Prediction   N   Y
##          N 951  90
##          Y 224 293
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         951                         224
## 2   Y                          90                         293
##           Reference
## Prediction    N    Y
##          N 1076  185
##          Y   99  198
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1076                          99
## 2   Y                         185                         198
##           Reference
## Prediction    N    Y
##          N 1076  185
##          Y   99  198
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1076                          99
## 2   Y                         185                         198
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1147                          28
## 2   Y                         276                         107
##           Reference
## Prediction    N    Y
##          N 1147  276
##          Y   28  107
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1147                          28
## 2   Y                         276                         107
##           Reference
## Prediction    N    Y
##          N 1168  323
##          Y    7   60
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1168                           7
## 2   Y                         323                          60
##           Reference
## Prediction    N    Y
##          N 1173  376
##          Y    2    7
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1173                           2
## 2   Y                         376                           7
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                        1175                           0
## 2   Y                         383                           0
##    threshold    f.score
## 1        0.0 0.39464194
## 2        0.1 0.53720765
## 3        0.2 0.63378995
## 4        0.3 0.65111111
## 5        0.4 0.58235294
## 6        0.5 0.58235294
## 7        0.6 0.41312741
## 8        0.7 0.41312741
## 9        0.8 0.26666667
## 10       0.9 0.03571429
## 11       1.0 0.00000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-24.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         951                         224
## 2   Y                          90                         293
##           Reference
## Prediction   N   Y
##          N 951  90
##          Y 224 293
##   isB isB.predict.Max.cor.Y.glm.N isB.predict.Max.cor.Y.glm.Y
## 1   N                         951                         224
## 2   Y                          90                         293
##          Prediction
## Reference   N   Y
##         N 951 224
##         Y  90 293
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.984596e-01   5.137918e-01   7.776652e-01   8.181225e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.940093e-05   6.113657e-14 
##        model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.glm          glm yedge               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.882                 0.026   0.8636631
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.6302596         0.814504
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7122496             0.7566947     0.4429064   0.8711827
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3       0.6511111        0.7984596
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.7776652             0.8181225     0.5137918    1197.518
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.003099694      0.02354958
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
## [1] "fitting model: Interact.High.cor.y.glm"
## [1] "    indep_vars: yedge, yedge:xbox, yedge:ybox, yedge:xy2bar, yedge:width, yedge:height"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-25.png) ![](Letter_Recognition_B_files/figure-html/fit.models-26.png) ![](Letter_Recognition_B_files/figure-html/fit.models-27.png) ![](Letter_Recognition_B_files/figure-html/fit.models-28.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -3.13342  -0.49652  -0.21154  -0.04102   3.10873  
## 
## Coefficients:
##                 Estimate Std. Error z value Pr(>|z|)    
## (Intercept)    -8.360356   0.424173 -19.710  < 2e-16 ***
## yedge           2.548543   0.156681  16.266  < 2e-16 ***
## `yedge:xbox`   -0.061580   0.017042  -3.613 0.000302 ***
## `yedge:ybox`    0.028265   0.009929   2.847 0.004417 ** 
## `yedge:xy2bar` -0.087918   0.010514  -8.362  < 2e-16 ***
## `yedge:width`   0.002120   0.013998   0.151 0.879605    
## `yedge:height` -0.096225   0.013681  -7.034 2.01e-12 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1737.81  on 1557  degrees of freedom
## Residual deviance:  952.58  on 1551  degrees of freedom
## AIC: 966.58
## 
## Number of Fisher Scoring iterations: 6
```

![](Letter_Recognition_B_files/figure-html/fit.models-29.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction   N   Y
##          N 748  16
##          Y 427 367
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                   748
## 2   Y                                    16
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   427
## 2                                   367
##           Reference
## Prediction   N   Y
##          N 959  53
##          Y 216 330
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                   959
## 2   Y                                    53
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   216
## 2                                   330
##           Reference
## Prediction    N    Y
##          N 1033   86
##          Y  142  297
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1033
## 2   Y                                    86
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   142
## 2                                   297
##           Reference
## Prediction    N    Y
##          N 1082  112
##          Y   93  271
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1082
## 2   Y                                   112
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    93
## 2                                   271
##           Reference
## Prediction    N    Y
##          N 1111  134
##          Y   64  249
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1111
## 2   Y                                   134
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    64
## 2                                   249
##           Reference
## Prediction    N    Y
##          N 1131  159
##          Y   44  224
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1131
## 2   Y                                   159
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    44
## 2                                   224
##           Reference
## Prediction    N    Y
##          N 1142  194
##          Y   33  189
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1142
## 2   Y                                   194
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    33
## 2                                   189
##           Reference
## Prediction    N    Y
##          N 1158  242
##          Y   17  141
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1158
## 2   Y                                   242
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    17
## 2                                   141
##           Reference
## Prediction    N    Y
##          N 1166  290
##          Y    9   93
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1166
## 2   Y                                   290
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                     9
## 2                                    93
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6236194
## 3        0.2 0.7104413
## 4        0.3 0.7226277
## 5        0.4 0.7255689
## 6        0.5 0.7155172
## 7        0.6 0.6881720
## 8        0.7 0.6247934
## 9        0.8 0.5212569
## 10       0.9 0.3835052
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-30.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1082
## 2   Y                                   112
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    93
## 2                                   271
##           Reference
## Prediction    N    Y
##          N 1082  112
##          Y   93  271
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1082
## 2   Y                                   112
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    93
## 2                                   271
##          Prediction
## Reference    N    Y
##         N 1082   93
##         Y  112  271
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.684211e-01   6.391082e-01   8.506195e-01   8.848207e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   3.209554e-29   2.086904e-01
```

![](Letter_Recognition_B_files/figure-html/fit.models-31.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                     0
## 2   Y                                     0
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                  1175
## 2                                   383
##           Reference
## Prediction   N   Y
##          N 755  18
##          Y 420 365
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                   755
## 2   Y                                    18
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   420
## 2                                   365
##           Reference
## Prediction   N   Y
##          N 946  56
##          Y 229 327
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                   946
## 2   Y                                    56
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   229
## 2                                   327
##           Reference
## Prediction    N    Y
##          N 1027   80
##          Y  148  303
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1027
## 2   Y                                    80
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   148
## 2                                   303
##           Reference
## Prediction    N    Y
##          N 1074  104
##          Y  101  279
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1074
## 2   Y                                   104
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                   101
## 2                                   279
##           Reference
## Prediction    N    Y
##          N 1108  123
##          Y   67  260
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1108
## 2   Y                                   123
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    67
## 2                                   260
##           Reference
## Prediction    N    Y
##          N 1131  148
##          Y   44  235
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1131
## 2   Y                                   148
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    44
## 2                                   235
##           Reference
## Prediction    N    Y
##          N 1149  186
##          Y   26  197
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1149
## 2   Y                                   186
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    26
## 2                                   197
##           Reference
## Prediction    N    Y
##          N 1161  230
##          Y   14  153
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1161
## 2   Y                                   230
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    14
## 2                                   153
##           Reference
## Prediction    N    Y
##          N 1171  278
##          Y    4  105
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1171
## 2   Y                                   278
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                     4
## 2                                   105
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1175
## 2   Y                                   383
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6250000
## 3        0.2 0.6964856
## 4        0.3 0.7266187
## 5        0.4 0.7313237
## 6        0.5 0.7323944
## 7        0.6 0.7099698
## 8        0.7 0.6501650
## 9        0.8 0.5563636
## 10       0.9 0.4268293
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-32.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1108
## 2   Y                                   123
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    67
## 2                                   260
##           Reference
## Prediction    N    Y
##          N 1108  123
##          Y   67  260
##   isB isB.predict.Interact.High.cor.y.glm.N
## 1   N                                  1108
## 2   Y                                   123
##   isB.predict.Interact.High.cor.y.glm.Y
## 1                                    67
## 2                                   260
##          Prediction
## Reference    N    Y
##         N 1108   67
##         Y  123  260
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.780488e-01   6.540602e-01   8.607636e-01   8.938943e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.658453e-34   6.604005e-05 
##                  model_id model_method
## 1 Interact.High.cor.y.glm          glm
##                                                                    feats
## 1 yedge, yedge:xbox, yedge:ybox, yedge:xy2bar, yedge:width, yedge:height
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.194                 0.053
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9169779                    0.4       0.7255689        0.8697026
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8506195             0.8848207      0.624709   0.9226987
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5       0.7323944        0.8780488
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.8607636             0.8938943     0.6540602    966.5847
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.00307697      0.02075419
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
## [1] "fitting model: Low.cor.X.glm"
## [1] "    indep_vars: yedge, y2bar, x2ybar, yedgexcor, onpix, x2bar, xedge, xbar, xedgeycor, .rnorm, ybar, xybar"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-33.png) ![](Letter_Recognition_B_files/figure-html/fit.models-34.png) ![](Letter_Recognition_B_files/figure-html/fit.models-35.png) ![](Letter_Recognition_B_files/figure-html/fit.models-36.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.9561  -0.2370  -0.0485  -0.0011   3.7809  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -18.97689    1.92862  -9.840  < 2e-16 ***
## yedge         1.69771    0.12975  13.085  < 2e-16 ***
## y2bar         0.96574    0.11590   8.332  < 2e-16 ***
## x2ybar        0.82737    0.11611   7.126 1.04e-12 ***
## yedgexcor     0.36106    0.07109   5.079 3.80e-07 ***
## onpix        -0.44181    0.07404  -5.968 2.41e-09 ***
## x2bar        -0.68295    0.10245  -6.666 2.62e-11 ***
## xedge        -0.50900    0.08727  -5.833 5.46e-09 ***
## xbar          0.75481    0.12150   6.212 5.22e-10 ***
## xedgeycor     0.28874    0.09716   2.972 0.002960 ** 
## .rnorm        0.02476    0.10310   0.240 0.810239    
## ybar         -0.27872    0.09895  -2.817 0.004851 ** 
## xybar        -0.27799    0.07194  -3.864 0.000111 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1737.81  on 1557  degrees of freedom
## Residual deviance:  620.28  on 1545  degrees of freedom
## AIC: 646.28
## 
## Number of Fisher Scoring iterations: 7
```

![](Letter_Recognition_B_files/figure-html/fit.models-37.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                           0                        1175
## 2   Y                           0                         383
##           Reference
## Prediction   N   Y
##          N 942  13
##          Y 233 370
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                         942                         233
## 2   Y                          13                         370
##           Reference
## Prediction    N    Y
##          N 1024   29
##          Y  151  354
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1024                         151
## 2   Y                          29                         354
##           Reference
## Prediction    N    Y
##          N 1068   38
##          Y  107  345
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1068                         107
## 2   Y                          38                         345
##           Reference
## Prediction    N    Y
##          N 1101   49
##          Y   74  334
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1101                          74
## 2   Y                          49                         334
##           Reference
## Prediction    N    Y
##          N 1122   70
##          Y   53  313
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1122                          53
## 2   Y                          70                         313
##           Reference
## Prediction    N    Y
##          N 1133   95
##          Y   42  288
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1133                          42
## 2   Y                          95                         288
##           Reference
## Prediction    N    Y
##          N 1145  119
##          Y   30  264
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1145                          30
## 2   Y                         119                         264
##           Reference
## Prediction    N    Y
##          N 1156  157
##          Y   19  226
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1156                          19
## 2   Y                         157                         226
##           Reference
## Prediction    N    Y
##          N 1168  219
##          Y    7  164
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1168                           7
## 2   Y                         219                         164
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1175                           0
## 2   Y                         383                           0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.7505071
## 3        0.2 0.7972973
## 4        0.3 0.8263473
## 5        0.4 0.8445006
## 6        0.5 0.8357810
## 7        0.6 0.8078541
## 8        0.7 0.7799114
## 9        0.8 0.7197452
## 10       0.9 0.5920578
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-38.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1101                          74
## 2   Y                          49                         334
##           Reference
## Prediction    N    Y
##          N 1101   49
##          Y   74  334
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1101                          74
## 2   Y                          49                         334
##          Prediction
## Reference    N    Y
##         N 1101   74
##         Y   49  334
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.210526e-01   7.916682e-01   9.065333e-01   9.339600e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   6.344830e-66   3.046380e-02
```

![](Letter_Recognition_B_files/figure-html/fit.models-39.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                           0                        1175
## 2   Y                           0                         383
##           Reference
## Prediction   N   Y
##          N 935  11
##          Y 240 372
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                         935                         240
## 2   Y                          11                         372
##           Reference
## Prediction    N    Y
##          N 1014   19
##          Y  161  364
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1014                         161
## 2   Y                          19                         364
##           Reference
## Prediction    N    Y
##          N 1065   33
##          Y  110  350
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1065                         110
## 2   Y                          33                         350
##           Reference
## Prediction    N    Y
##          N 1093   40
##          Y   82  343
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1093                          82
## 2   Y                          40                         343
##           Reference
## Prediction    N    Y
##          N 1114   59
##          Y   61  324
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1114                          61
## 2   Y                          59                         324
##           Reference
## Prediction    N    Y
##          N 1134   79
##          Y   41  304
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1134                          41
## 2   Y                          79                         304
##           Reference
## Prediction    N    Y
##          N 1145  114
##          Y   30  269
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1145                          30
## 2   Y                         114                         269
##           Reference
## Prediction    N    Y
##          N 1161  146
##          Y   14  237
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1161                          14
## 2   Y                         146                         237
##           Reference
## Prediction    N    Y
##          N 1169  206
##          Y    6  177
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1169                           6
## 2   Y                         206                         177
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1175                           0
## 2   Y                         383                           0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.7477387
## 3        0.2 0.8017621
## 4        0.3 0.8303677
## 5        0.4 0.8490099
## 6        0.5 0.8437500
## 7        0.6 0.8351648
## 8        0.7 0.7888563
## 9        0.8 0.7476341
## 10       0.9 0.6254417
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-40.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1093                          82
## 2   Y                          40                         343
##           Reference
## Prediction    N    Y
##          N 1093   40
##          Y   82  343
##   isB isB.predict.Low.cor.X.glm.N isB.predict.Low.cor.X.glm.Y
## 1   N                        1093                          82
## 2   Y                          40                         343
##          Prediction
## Reference    N    Y
##         N 1093   82
##         Y   40  343
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.216945e-01   7.963429e-01   9.072236e-01   9.345508e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.662089e-66   2.056560e-04 
##        model_id model_method
## 1 Low.cor.X.glm          glm
##                                                                                        feats
## 1 yedge, y2bar, x2ybar, yedgexcor, onpix, x2bar, xedge, xbar, xedgeycor, .rnorm, ybar, xybar
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.339                 0.087
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9663396                    0.4       0.8445006        0.9152809
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9065333               0.93396      0.768761   0.9694195
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.8490099        0.9216945
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.9072236             0.9345508     0.7963429    646.2841
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.006848514      0.01679136
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
## [1] "iterating over method:glm"
## [1] "fitting model: All.X.glm"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

![](Letter_Recognition_B_files/figure-html/fit.models-41.png) ![](Letter_Recognition_B_files/figure-html/fit.models-42.png) ![](Letter_Recognition_B_files/figure-html/fit.models-43.png) ![](Letter_Recognition_B_files/figure-html/fit.models-44.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -3.2401  -0.1139  -0.0063   0.0000   3.8137  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -19.15328    3.45103  -5.550 2.86e-08 ***
## xbox          0.14563    0.19497   0.747 0.455083    
## ybox          0.01717    0.12826   0.134 0.893511    
## width        -1.93545    0.25065  -7.722 1.15e-14 ***
## height       -0.94306    0.20104  -4.691 2.72e-06 ***
## onpix         1.65453    0.22596   7.322 2.44e-13 ***
## xbar          0.79913    0.17143   4.662 3.14e-06 ***
## ybar         -0.48814    0.15632  -3.123 0.001792 ** 
## x2bar        -0.51591    0.13951  -3.698 0.000217 ***
## y2bar         1.69059    0.17957   9.414  < 2e-16 ***
## xybar         0.42731    0.12673   3.372 0.000747 ***
## x2ybar        0.70065    0.17612   3.978 6.94e-05 ***
## xy2bar       -0.50257    0.15485  -3.245 0.001173 ** 
## xedge        -0.23773    0.12855  -1.849 0.064414 .  
## xedgeycor     0.01703    0.13896   0.123 0.902488    
## yedge         1.87108    0.16830  11.118  < 2e-16 ***
## yedgexcor     0.39414    0.08831   4.463 8.08e-06 ***
## .rnorm       -0.07722    0.12681  -0.609 0.542563    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1737.81  on 1557  degrees of freedom
## Residual deviance:  414.82  on 1540  degrees of freedom
## AIC: 450.82
## 
## Number of Fisher Scoring iterations: 9
```

![](Letter_Recognition_B_files/figure-html/fit.models-45.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                       0                    1175
## 2   Y                       0                     383
##           Reference
## Prediction    N    Y
##          N 1027   14
##          Y  148  369
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1027                     148
## 2   Y                      14                     369
##           Reference
## Prediction    N    Y
##          N 1086   22
##          Y   89  361
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1086                      89
## 2   Y                      22                     361
##           Reference
## Prediction    N    Y
##          N 1112   25
##          Y   63  358
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1112                      63
## 2   Y                      25                     358
##           Reference
## Prediction    N    Y
##          N 1131   29
##          Y   44  354
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1131                      44
## 2   Y                      29                     354
##           Reference
## Prediction    N    Y
##          N 1143   38
##          Y   32  345
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1143                      32
## 2   Y                      38                     345
##           Reference
## Prediction    N    Y
##          N 1148   51
##          Y   27  332
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1148                      27
## 2   Y                      51                     332
##           Reference
## Prediction    N    Y
##          N 1160   64
##          Y   15  319
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1160                      15
## 2   Y                      64                     319
##           Reference
## Prediction    N    Y
##          N 1165   89
##          Y   10  294
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1165                      10
## 2   Y                      89                     294
##           Reference
## Prediction    N    Y
##          N 1174  132
##          Y    1  251
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1174                       1
## 2   Y                     132                     251
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1175                       0
## 2   Y                     383                       0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.8200000
## 3        0.2 0.8667467
## 4        0.3 0.8905473
## 5        0.4 0.9065301
## 6        0.5 0.9078947
## 7        0.6 0.8948787
## 8        0.7 0.8898187
## 9        0.8 0.8558952
## 10       0.9 0.7905512
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-46.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1143                      32
## 2   Y                      38                     345
##           Reference
## Prediction    N    Y
##          N 1143   38
##          Y   32  345
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1143                      32
## 2   Y                      38                     345
##          Prediction
## Reference    N    Y
##         N 1143   32
##         Y   38  345
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.550706e-01   8.781858e-01   9.435727e-01   9.648107e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##  6.390251e-103   5.500973e-01
```

![](Letter_Recognition_B_files/figure-html/fit.models-47.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                       0                    1175
## 2   Y                       0                     383
##           Reference
## Prediction    N    Y
##          N 1001   12
##          Y  174  371
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1001                     174
## 2   Y                      12                     371
##           Reference
## Prediction    N    Y
##          N 1055   18
##          Y  120  365
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1055                     120
## 2   Y                      18                     365
##           Reference
## Prediction    N    Y
##          N 1090   21
##          Y   85  362
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1090                      85
## 2   Y                      21                     362
##           Reference
## Prediction    N    Y
##          N 1121   30
##          Y   54  353
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1121                      54
## 2   Y                      30                     353
##           Reference
## Prediction    N    Y
##          N 1131   35
##          Y   44  348
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1131                      44
## 2   Y                      35                     348
##           Reference
## Prediction    N    Y
##          N 1141   44
##          Y   34  339
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1141                      34
## 2   Y                      44                     339
##           Reference
## Prediction    N    Y
##          N 1154   55
##          Y   21  328
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1154                      21
## 2   Y                      55                     328
##           Reference
## Prediction    N    Y
##          N 1163   77
##          Y   12  306
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1163                      12
## 2   Y                      77                     306
##           Reference
## Prediction    N    Y
##          N 1168  126
##          Y    7  257
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1168                       7
## 2   Y                     126                     257
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1175                       0
## 2   Y                     383                       0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.7995690
## 3        0.2 0.8410138
## 4        0.3 0.8722892
## 5        0.4 0.8936709
## 6        0.5 0.8980645
## 7        0.6 0.8968254
## 8        0.7 0.8961749
## 9        0.8 0.8730385
## 10       0.9 0.7944359
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-48.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1131                      44
## 2   Y                      35                     348
##           Reference
## Prediction    N    Y
##          N 1131   35
##          Y   44  348
##   isB isB.predict.All.X.glm.N isB.predict.All.X.glm.Y
## 1   N                    1131                      44
## 2   Y                      35                     348
##          Prediction
## Reference    N    Y
##         N 1131   44
##         Y   35  348
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.492940e-01   8.643243e-01   9.372030e-01   9.596531e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.269472e-95   3.680828e-01 
##    model_id model_method
## 1 All.X.glm          glm
##                                                                                                                           feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.482                 0.118
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.984032                    0.5       0.9078947        0.9467356
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9435727             0.9648107     0.8564668   0.9833165
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5       0.8980645         0.949294
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1              0.937203             0.9596531     0.8643243    450.8178
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01551523      0.04095663
## [1] "iterating over method:rpart"
## [1] "fitting model: All.X.rpart"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor"
## + Fold1: cp=0.02872 
## - Fold1: cp=0.02872 
## + Fold2: cp=0.02872 
## - Fold2: cp=0.02872 
## + Fold3: cp=0.02872 
## - Fold3: cp=0.02872 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0287 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## cp
```

![](Letter_Recognition_B_files/figure-html/fit.models-49.png) ![](Letter_Recognition_B_files/figure-html/fit.models-50.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##           CP nsplit rel error
## 1 0.20887728      0 1.0000000
## 2 0.07963446      2 0.5822454
## 3 0.02872063      4 0.4229765
## 
## Variable importance
##     yedge     xedge     onpix    xy2bar     y2bar yedgexcor     x2bar 
##        18        14        11        11        11        11         8 
## xedgeycor     width      ybar      xbox      xbar    height    x2ybar 
##         4         3         3         3         2         2         1 
## 
## Node number 1: 1558 observations,    complexity param=0.2088773
##   predicted class=N  expected loss=0.245828  P(node) =1
##     class counts:  1175   383
##    probabilities: 0.754 0.246 
##   left son=2 (824 obs) right son=3 (734 obs)
##   Primary splits:
##       yedge     < 4.5  to the left,  improve=151.64140, (0 missing)
##       y2bar     < 4.5  to the left,  improve=144.56850, (0 missing)
##       yedgexcor < 8.5  to the left,  improve= 75.80293, (0 missing)
##       xedgeycor < 7.5  to the left,  improve= 69.62928, (0 missing)
##       x2ybar    < 4.5  to the left,  improve= 67.42586, (0 missing)
##   Surrogate splits:
##       onpix     < 3.5  to the left,  agree=0.745, adj=0.458, (0 split)
##       yedgexcor < 8.5  to the left,  agree=0.702, adj=0.368, (0 split)
##       y2bar     < 3.5  to the left,  agree=0.697, adj=0.357, (0 split)
##       xedge     < 2.5  to the left,  agree=0.690, adj=0.342, (0 split)
##       x2bar     < 3.5  to the left,  agree=0.681, adj=0.323, (0 split)
## 
## Node number 2: 824 observations
##   predicted class=N  expected loss=0.03762136  P(node) =0.5288832
##     class counts:   793    31
##    probabilities: 0.962 0.038 
## 
## Node number 3: 734 observations,    complexity param=0.2088773
##   predicted class=N  expected loss=0.479564  P(node) =0.4711168
##     class counts:   382   352
##    probabilities: 0.520 0.480 
##   left son=6 (284 obs) right son=7 (450 obs)
##   Primary splits:
##       xy2bar    < 7.5  to the right, improve=91.38770, (0 missing)
##       y2bar     < 4.5  to the left,  improve=87.66077, (0 missing)
##       xedge     < 2.5  to the right, improve=80.74484, (0 missing)
##       ybar      < 8.5  to the right, improve=55.78698, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=48.50795, (0 missing)
##   Surrogate splits:
##       yedgexcor < 10.5 to the right, agree=0.717, adj=0.268, (0 split)
##       xedgeycor < 7.5  to the left,  agree=0.704, adj=0.236, (0 split)
##       xbar      < 5.5  to the left,  agree=0.689, adj=0.197, (0 split)
##       y2bar     < 4.5  to the left,  agree=0.683, adj=0.180, (0 split)
##       ybar      < 9.5  to the right, agree=0.681, adj=0.176, (0 split)
## 
## Node number 6: 284 observations
##   predicted class=N  expected loss=0.165493  P(node) =0.182285
##     class counts:   237    47
##    probabilities: 0.835 0.165 
## 
## Node number 7: 450 observations,    complexity param=0.07963446
##   predicted class=Y  expected loss=0.3222222  P(node) =0.2888318
##     class counts:   145   305
##    probabilities: 0.322 0.678 
##   left son=14 (258 obs) right son=15 (192 obs)
##   Primary splits:
##       xedge     < 2.5  to the right, improve=60.83850, (0 missing)
##       y2bar     < 4.5  to the left,  improve=51.64374, (0 missing)
##       xedgeycor < 9.5  to the right, improve=45.93312, (0 missing)
##       yedgexcor < 8.5  to the left,  improve=38.71605, (0 missing)
##       xy2bar    < 4.5  to the left,  improve=28.47955, (0 missing)
##   Surrogate splits:
##       width  < 5.5  to the right, agree=0.769, adj=0.458, (0 split)
##       onpix  < 4.5  to the right, agree=0.744, adj=0.401, (0 split)
##       xbox   < 3.5  to the right, agree=0.722, adj=0.349, (0 split)
##       x2bar  < 6.5  to the left,  agree=0.716, adj=0.333, (0 split)
##       height < 5.5  to the right, agree=0.698, adj=0.292, (0 split)
## 
## Node number 14: 258 observations,    complexity param=0.07963446
##   predicted class=N  expected loss=0.4534884  P(node) =0.1655969
##     class counts:   141   117
##    probabilities: 0.547 0.453 
##   left son=28 (115 obs) right son=29 (143 obs)
##   Primary splits:
##       y2bar     < 4.5  to the left,  improve=19.84870, (0 missing)
##       xedgeycor < 9.5  to the right, improve=15.28524, (0 missing)
##       xybar     < 11.5 to the right, improve=12.40320, (0 missing)
##       ybar      < 8.5  to the right, improve=12.18987, (0 missing)
##       xy2bar    < 4.5  to the left,  improve=11.89234, (0 missing)
##   Surrogate splits:
##       xedgeycor < 9.5  to the right, agree=0.764, adj=0.470, (0 split)
##       yedgexcor < 7.5  to the left,  agree=0.752, adj=0.443, (0 split)
##       xedge     < 3.5  to the right, agree=0.694, adj=0.313, (0 split)
##       x2ybar    < 6.5  to the right, agree=0.674, adj=0.270, (0 split)
##       ybar      < 8.5  to the right, agree=0.671, adj=0.261, (0 split)
## 
## Node number 15: 192 observations
##   predicted class=Y  expected loss=0.02083333  P(node) =0.1232349
##     class counts:     4   188
##    probabilities: 0.021 0.979 
## 
## Node number 28: 115 observations
##   predicted class=N  expected loss=0.2347826  P(node) =0.07381258
##     class counts:    88    27
##    probabilities: 0.765 0.235 
## 
## Node number 29: 143 observations
##   predicted class=Y  expected loss=0.3706294  P(node) =0.09178434
##     class counts:    53    90
##    probabilities: 0.371 0.629 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1558 383 N (0.75417202 0.24582798)  
##    2) yedge< 4.5 824  31 N (0.96237864 0.03762136) *
##    3) yedge>=4.5 734 352 N (0.52043597 0.47956403)  
##      6) xy2bar>=7.5 284  47 N (0.83450704 0.16549296) *
##      7) xy2bar< 7.5 450 145 Y (0.32222222 0.67777778)  
##       14) xedge>=2.5 258 117 N (0.54651163 0.45348837)  
##         28) y2bar< 4.5 115  27 N (0.76521739 0.23478261) *
##         29) y2bar>=4.5 143  53 Y (0.37062937 0.62937063) *
##       15) xedge< 2.5 192   4 Y (0.02083333 0.97916667) *
```

![](Letter_Recognition_B_files/figure-html/fit.models-51.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                         0                      1175
## 2   Y                         0                       383
##           Reference
## Prediction   N   Y
##          N 793  31
##          Y 382 352
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                       793                       382
## 2   Y                        31                       352
##           Reference
## Prediction    N    Y
##          N 1030   78
##          Y  145  305
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1030                       145
## 2   Y                        78                       305
##           Reference
## Prediction    N    Y
##          N 1118  105
##          Y   57  278
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##           Reference
## Prediction    N    Y
##          N 1118  105
##          Y   57  278
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##           Reference
## Prediction    N    Y
##          N 1118  105
##          Y   57  278
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##           Reference
## Prediction    N    Y
##          N 1118  105
##          Y   57  278
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##           Reference
## Prediction    N    Y
##          N 1171  195
##          Y    4  188
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1171                         4
## 2   Y                       195                       188
##           Reference
## Prediction    N    Y
##          N 1171  195
##          Y    4  188
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1171                         4
## 2   Y                       195                       188
##           Reference
## Prediction    N    Y
##          N 1171  195
##          Y    4  188
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1171                         4
## 2   Y                       195                       188
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1175                         0
## 2   Y                       383                         0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6302596
## 3        0.2 0.7322929
## 4        0.3 0.7743733
## 5        0.4 0.7743733
## 6        0.5 0.7743733
## 7        0.6 0.7743733
## 8        0.7 0.6539130
## 9        0.8 0.6539130
## 10       0.9 0.6539130
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-52.png) 

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.fit"
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##           Reference
## Prediction    N    Y
##          N 1118  105
##          Y   57  278
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1118                        57
## 2   Y                       105                       278
##          Prediction
## Reference    N    Y
##         N 1118   57
##         Y  105  278
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.960205e-01   7.072088e-01   8.797895e-01   9.107407e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   5.746554e-46   2.219130e-04
```

![](Letter_Recognition_B_files/figure-html/fit.models-53.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                         0                      1175
## 2   Y                         0                       383
##           Reference
## Prediction   N   Y
##          N 810  36
##          Y 365 347
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                       810                       365
## 2   Y                        36                       347
##           Reference
## Prediction    N    Y
##          N 1018   75
##          Y  157  308
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1018                       157
## 2   Y                        75                       308
##           Reference
## Prediction    N    Y
##          N 1101   96
##          Y   74  287
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##           Reference
## Prediction    N    Y
##          N 1101   96
##          Y   74  287
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##           Reference
## Prediction    N    Y
##          N 1101   96
##          Y   74  287
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##           Reference
## Prediction    N    Y
##          N 1101   96
##          Y   74  287
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##           Reference
## Prediction    N    Y
##          N 1161  180
##          Y   14  203
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1161                        14
## 2   Y                       180                       203
##           Reference
## Prediction    N    Y
##          N 1161  180
##          Y   14  203
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1161                        14
## 2   Y                       180                       203
##           Reference
## Prediction    N    Y
##          N 1161  180
##          Y   14  203
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1161                        14
## 2   Y                       180                       203
##           Reference
## Prediction    N    Y
##          N 1175  383
##          Y    0    0
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1175                         0
## 2   Y                       383                         0
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.6337900
## 3        0.2 0.7264151
## 4        0.3 0.7715054
## 5        0.4 0.7715054
## 6        0.5 0.7715054
## 7        0.6 0.7715054
## 8        0.7 0.6766667
## 9        0.8 0.6766667
## 10       0.9 0.6766667
## 11       1.0 0.0000000
```

![](Letter_Recognition_B_files/figure-html/fit.models-54.png) 

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.OOB"
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##           Reference
## Prediction    N    Y
##          N 1101   96
##          Y   74  287
##   isB isB.predict.All.X.rpart.N isB.predict.All.X.rpart.Y
## 1   N                      1101                        74
## 2   Y                        96                       287
##          Prediction
## Reference    N    Y
##         N 1101   74
##         Y   96  287
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.908858e-01   6.999182e-01   8.743405e-01   9.059406e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   1.809030e-42   1.072612e-01 
##      model_id model_method
## 1 All.X.rpart        rpart
##                                                                                                                   feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      1.557                 0.093
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9058586                    0.6       0.7743733        0.8934502
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8797895             0.9107407     0.7024631   0.8990245
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.6       0.7715054        0.8908858
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8743405             0.9059406     0.6999182
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.006265633      0.02004643
## [1] "fitting model: All.X.cp.0.rpart"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor"
## Fitting cp = 0 on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.models-55.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1558 
## 
##             CP nsplit rel error
## 1  0.208877285      0 1.0000000
## 2  0.079634465      2 0.5822454
## 3  0.028720627      4 0.4229765
## 4  0.024804178      5 0.3942559
## 5  0.023498695      7 0.3446475
## 6  0.020887728      8 0.3211488
## 7  0.013054830     11 0.2584856
## 8  0.010443864     15 0.2062663
## 9  0.005221932     18 0.1749347
## 10 0.002610966     19 0.1697128
## 11 0.000000000     20 0.1671018
## 
## Variable importance
##     yedge     xedge    xy2bar yedgexcor     onpix     x2bar     y2bar 
##        15        12        11         9         8         8         8 
## xedgeycor     xybar      ybar      xbox     width      xbar    height 
##         6         5         5         4         3         3         2 
##    x2ybar      ybox 
##         1         1 
## 
## Node number 1: 1558 observations,    complexity param=0.2088773
##   predicted class=N  expected loss=0.245828  P(node) =1
##     class counts:  1175   383
##    probabilities: 0.754 0.246 
##   left son=2 (824 obs) right son=3 (734 obs)
##   Primary splits:
##       yedge     < 4.5  to the left,  improve=151.64140, (0 missing)
##       y2bar     < 4.5  to the left,  improve=144.56850, (0 missing)
##       yedgexcor < 8.5  to the left,  improve= 75.80293, (0 missing)
##       xedgeycor < 7.5  to the left,  improve= 69.62928, (0 missing)
##       x2ybar    < 4.5  to the left,  improve= 67.42586, (0 missing)
##   Surrogate splits:
##       onpix     < 3.5  to the left,  agree=0.745, adj=0.458, (0 split)
##       yedgexcor < 8.5  to the left,  agree=0.702, adj=0.368, (0 split)
##       y2bar     < 3.5  to the left,  agree=0.697, adj=0.357, (0 split)
##       xedge     < 2.5  to the left,  agree=0.690, adj=0.342, (0 split)
##       x2bar     < 3.5  to the left,  agree=0.681, adj=0.323, (0 split)
## 
## Node number 2: 824 observations,    complexity param=0.02088773
##   predicted class=N  expected loss=0.03762136  P(node) =0.5288832
##     class counts:   793    31
##    probabilities: 0.962 0.038 
##   left son=4 (607 obs) right son=5 (217 obs)
##   Primary splits:
##       yedgexcor < 8.5  to the left,  improve=5.431807, (0 missing)
##       y2bar     < 4.5  to the left,  improve=5.128119, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=3.006365, (0 missing)
##       height    < 2.5  to the right, improve=2.536471, (0 missing)
##       xbar      < 7.5  to the left,  improve=2.073273, (0 missing)
##   Surrogate splits:
##       y2bar     < 4.5  to the left,  agree=0.767, adj=0.115, (0 split)
##       xbar      < 11.5 to the left,  agree=0.748, adj=0.041, (0 split)
##       xedgeycor < 3.5  to the right, agree=0.745, adj=0.032, (0 split)
##       xy2bar    < 10.5 to the left,  agree=0.740, adj=0.014, (0 split)
##       width     < 1.5  to the right, agree=0.739, adj=0.009, (0 split)
## 
## Node number 3: 734 observations,    complexity param=0.2088773
##   predicted class=N  expected loss=0.479564  P(node) =0.4711168
##     class counts:   382   352
##    probabilities: 0.520 0.480 
##   left son=6 (284 obs) right son=7 (450 obs)
##   Primary splits:
##       xy2bar    < 7.5  to the right, improve=91.38770, (0 missing)
##       y2bar     < 4.5  to the left,  improve=87.66077, (0 missing)
##       xedge     < 2.5  to the right, improve=80.74484, (0 missing)
##       ybar      < 8.5  to the right, improve=55.78698, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=48.50795, (0 missing)
##   Surrogate splits:
##       yedgexcor < 10.5 to the right, agree=0.717, adj=0.268, (0 split)
##       xedgeycor < 7.5  to the left,  agree=0.704, adj=0.236, (0 split)
##       xbar      < 5.5  to the left,  agree=0.689, adj=0.197, (0 split)
##       y2bar     < 4.5  to the left,  agree=0.683, adj=0.180, (0 split)
##       ybar      < 9.5  to the right, agree=0.681, adj=0.176, (0 split)
## 
## Node number 4: 607 observations
##   predicted class=N  expected loss=0.003294893  P(node) =0.3896021
##     class counts:   605     2
##    probabilities: 0.997 0.003 
## 
## Node number 5: 217 observations,    complexity param=0.02088773
##   predicted class=N  expected loss=0.1336406  P(node) =0.1392811
##     class counts:   188    29
##    probabilities: 0.866 0.134 
##   left son=10 (152 obs) right son=11 (65 obs)
##   Primary splits:
##       xedgeycor < 7.5  to the left,  improve=18.125770, (0 missing)
##       x2ybar    < 4.5  to the left,  improve= 6.838717, (0 missing)
##       xedge     < 2.5  to the right, improve= 5.287608, (0 missing)
##       x2bar     < 3.5  to the right, improve= 5.228359, (0 missing)
##       height    < 2.5  to the right, improve= 4.672330, (0 missing)
##   Surrogate splits:
##       xy2bar < 5.5  to the right, agree=0.834, adj=0.446, (0 split)
##       xedge  < 1.5  to the right, agree=0.825, adj=0.415, (0 split)
##       xybar  < 11.5 to the left,  agree=0.793, adj=0.308, (0 split)
##       x2bar  < 2.5  to the right, agree=0.774, adj=0.246, (0 split)
##       ybar   < 10.5 to the left,  agree=0.728, adj=0.092, (0 split)
## 
## Node number 6: 284 observations,    complexity param=0.02480418
##   predicted class=N  expected loss=0.165493  P(node) =0.182285
##     class counts:   237    47
##    probabilities: 0.835 0.165 
##   left son=12 (244 obs) right son=13 (40 obs)
##   Primary splits:
##       yedge  < 7.5  to the left,  improve=15.614970, (0 missing)
##       xedge  < 4.5  to the left,  improve=14.052250, (0 missing)
##       x2bar  < 5.5  to the right, improve=11.851050, (0 missing)
##       xbar   < 6.5  to the left,  improve= 9.724981, (0 missing)
##       x2ybar < 5.5  to the left,  improve= 8.775679, (0 missing)
##   Surrogate splits:
##       yedgexcor < 3.5  to the right, agree=0.894, adj=0.250, (0 split)
##       onpix     < 7.5  to the left,  agree=0.877, adj=0.125, (0 split)
##       xedgeycor < 9.5  to the left,  agree=0.866, adj=0.050, (0 split)
## 
## Node number 7: 450 observations,    complexity param=0.07963446
##   predicted class=Y  expected loss=0.3222222  P(node) =0.2888318
##     class counts:   145   305
##    probabilities: 0.322 0.678 
##   left son=14 (258 obs) right son=15 (192 obs)
##   Primary splits:
##       xedge     < 2.5  to the right, improve=60.83850, (0 missing)
##       y2bar     < 4.5  to the left,  improve=51.64374, (0 missing)
##       xedgeycor < 9.5  to the right, improve=45.93312, (0 missing)
##       yedgexcor < 8.5  to the left,  improve=38.71605, (0 missing)
##       xy2bar    < 4.5  to the left,  improve=28.47955, (0 missing)
##   Surrogate splits:
##       width  < 5.5  to the right, agree=0.769, adj=0.458, (0 split)
##       onpix  < 4.5  to the right, agree=0.744, adj=0.401, (0 split)
##       xbox   < 3.5  to the right, agree=0.722, adj=0.349, (0 split)
##       x2bar  < 6.5  to the left,  agree=0.716, adj=0.333, (0 split)
##       height < 5.5  to the right, agree=0.698, adj=0.292, (0 split)
## 
## Node number 10: 152 observations
##   predicted class=N  expected loss=0  P(node) =0.09756098
##     class counts:   152     0
##    probabilities: 1.000 0.000 
## 
## Node number 11: 65 observations,    complexity param=0.02088773
##   predicted class=N  expected loss=0.4461538  P(node) =0.04172015
##     class counts:    36    29
##    probabilities: 0.554 0.446 
##   left son=22 (31 obs) right son=23 (34 obs)
##   Primary splits:
##       xedgeycor < 8.5  to the right, improve=23.59367, (0 missing)
##       xybar     < 10.5 to the right, improve=21.39063, (0 missing)
##       xy2bar    < 5.5  to the left,  improve=20.84530, (0 missing)
##       ybar      < 7.5  to the right, improve=17.74760, (0 missing)
##       xbox      < 3.5  to the right, improve=15.46031, (0 missing)
##   Surrogate splits:
##       xy2bar < 5.5  to the left,  agree=0.969, adj=0.935, (0 split)
##       xybar  < 10.5 to the right, agree=0.954, adj=0.903, (0 split)
##       ybar   < 7.5  to the right, agree=0.938, adj=0.871, (0 split)
##       xedge  < 1.5  to the left,  agree=0.831, adj=0.645, (0 split)
##       xbox   < 3.5  to the right, agree=0.785, adj=0.548, (0 split)
## 
## Node number 12: 244 observations,    complexity param=0.01044386
##   predicted class=N  expected loss=0.09836066  P(node) =0.156611
##     class counts:   220    24
##    probabilities: 0.902 0.098 
##   left son=24 (214 obs) right son=25 (30 obs)
##   Primary splits:
##       xybar < 9.5  to the left,  improve=7.676196, (0 missing)
##       xedge < 4.5  to the left,  improve=4.740269, (0 missing)
##       yedge < 6.5  to the left,  improve=3.908277, (0 missing)
##       ybar  < 7.5  to the right, improve=3.500911, (0 missing)
##       xbar  < 6.5  to the left,  improve=3.477142, (0 missing)
##   Surrogate splits:
##       xbar      < 8.5  to the left,  agree=0.934, adj=0.467, (0 split)
##       xy2bar    < 10.5 to the left,  agree=0.906, adj=0.233, (0 split)
##       xedgeycor < 3.5  to the right, agree=0.898, adj=0.167, (0 split)
##       xbox      < 9.5  to the left,  agree=0.893, adj=0.133, (0 split)
##       ybar      < 2.5  to the right, agree=0.889, adj=0.100, (0 split)
## 
## Node number 13: 40 observations,    complexity param=0.02480418
##   predicted class=Y  expected loss=0.425  P(node) =0.02567394
##     class counts:    17    23
##    probabilities: 0.425 0.575 
##   left son=26 (13 obs) right son=27 (27 obs)
##   Primary splits:
##       x2bar     < 5.5  to the right, improve=12.735190, (0 missing)
##       xedge     < 4.5  to the left,  improve= 7.679032, (0 missing)
##       xy2bar    < 8.5  to the right, improve= 7.111129, (0 missing)
##       yedgexcor < 3.5  to the left,  improve= 6.016667, (0 missing)
##       xybar     < 8.5  to the left,  improve= 3.612500, (0 missing)
##   Surrogate splits:
##       xedge     < 4.5  to the left,  agree=0.900, adj=0.692, (0 split)
##       yedgexcor < 3.5  to the left,  agree=0.875, adj=0.615, (0 split)
##       xy2bar    < 8.5  to the right, agree=0.800, adj=0.385, (0 split)
##       yedge     < 9.5  to the right, agree=0.775, adj=0.308, (0 split)
##       ybar      < 5.5  to the left,  agree=0.725, adj=0.154, (0 split)
## 
## Node number 14: 258 observations,    complexity param=0.07963446
##   predicted class=N  expected loss=0.4534884  P(node) =0.1655969
##     class counts:   141   117
##    probabilities: 0.547 0.453 
##   left son=28 (115 obs) right son=29 (143 obs)
##   Primary splits:
##       y2bar     < 4.5  to the left,  improve=19.84870, (0 missing)
##       xedgeycor < 9.5  to the right, improve=15.28524, (0 missing)
##       xybar     < 11.5 to the right, improve=12.40320, (0 missing)
##       ybar      < 8.5  to the right, improve=12.18987, (0 missing)
##       xy2bar    < 4.5  to the left,  improve=11.89234, (0 missing)
##   Surrogate splits:
##       xedgeycor < 9.5  to the right, agree=0.764, adj=0.470, (0 split)
##       yedgexcor < 7.5  to the left,  agree=0.752, adj=0.443, (0 split)
##       xedge     < 3.5  to the right, agree=0.694, adj=0.313, (0 split)
##       x2ybar    < 6.5  to the right, agree=0.674, adj=0.270, (0 split)
##       ybar      < 8.5  to the right, agree=0.671, adj=0.261, (0 split)
## 
## Node number 15: 192 observations
##   predicted class=Y  expected loss=0.02083333  P(node) =0.1232349
##     class counts:     4   188
##    probabilities: 0.021 0.979 
## 
## Node number 22: 31 observations
##   predicted class=N  expected loss=0  P(node) =0.0198973
##     class counts:    31     0
##    probabilities: 1.000 0.000 
## 
## Node number 23: 34 observations,    complexity param=0.002610966
##   predicted class=Y  expected loss=0.1470588  P(node) =0.02182285
##     class counts:     5    29
##    probabilities: 0.147 0.853 
##   left son=46 (9 obs) right son=47 (25 obs)
##   Primary splits:
##       ybox   < 5.5  to the right, improve=4.084967, (0 missing)
##       onpix  < 3.5  to the right, improve=4.084967, (0 missing)
##       width  < 4.5  to the right, improve=3.074866, (0 missing)
##       xbox   < 2.5  to the right, improve=2.696078, (0 missing)
##       height < 3.5  to the right, improve=2.696078, (0 missing)
##   Surrogate splits:
##       onpix  < 3.5  to the right, agree=1.000, adj=1.000, (0 split)
##       xbox   < 2.5  to the right, agree=0.912, adj=0.667, (0 split)
##       height < 3.5  to the right, agree=0.912, adj=0.667, (0 split)
##       width  < 4.5  to the right, agree=0.882, adj=0.556, (0 split)
##       ybar   < 5.5  to the left,  agree=0.882, adj=0.556, (0 split)
## 
## Node number 24: 214 observations
##   predicted class=N  expected loss=0.05140187  P(node) =0.1373556
##     class counts:   203    11
##    probabilities: 0.949 0.051 
## 
## Node number 25: 30 observations,    complexity param=0.01044386
##   predicted class=N  expected loss=0.4333333  P(node) =0.01925546
##     class counts:    17    13
##    probabilities: 0.567 0.433 
##   left son=50 (22 obs) right son=51 (8 obs)
##   Primary splits:
##       yedge     < 6.5  to the left,  improve=7.006061, (0 missing)
##       y2bar     < 4.5  to the left,  improve=6.522807, (0 missing)
##       xy2bar    < 10   to the right, improve=5.633333, (0 missing)
##       xedgeycor < 5.5  to the left,  improve=5.633333, (0 missing)
##       ybar      < 4.5  to the left,  improve=4.828571, (0 missing)
##   Surrogate splits:
##       xedgeycor < 6.5  to the left,  agree=0.800, adj=0.250, (0 split)
##       xbar      < 7.5  to the right, agree=0.767, adj=0.125, (0 split)
## 
## Node number 26: 13 observations
##   predicted class=N  expected loss=0  P(node) =0.008344031
##     class counts:    13     0
##    probabilities: 1.000 0.000 
## 
## Node number 27: 27 observations
##   predicted class=Y  expected loss=0.1481481  P(node) =0.01732991
##     class counts:     4    23
##    probabilities: 0.148 0.852 
## 
## Node number 28: 115 observations,    complexity param=0.01305483
##   predicted class=N  expected loss=0.2347826  P(node) =0.07381258
##     class counts:    88    27
##    probabilities: 0.765 0.235 
##   left son=56 (56 obs) right son=57 (59 obs)
##   Primary splits:
##       xybar  < 7.5  to the right, improve=10.272710, (0 missing)
##       yedge  < 7.5  to the left,  improve= 5.309009, (0 missing)
##       xy2bar < 6.5  to the left,  improve= 4.835400, (0 missing)
##       onpix  < 10.5 to the left,  improve= 4.564730, (0 missing)
##       ybar   < 8.5  to the right, improve= 4.188406, (0 missing)
##   Surrogate splits:
##       xy2bar    < 6.5  to the left,  agree=0.748, adj=0.482, (0 split)
##       ybar      < 8.5  to the right, agree=0.722, adj=0.429, (0 split)
##       xbox      < 5.5  to the right, agree=0.713, adj=0.411, (0 split)
##       xedgeycor < 8.5  to the right, agree=0.652, adj=0.286, (0 split)
##       ybox      < 11.5 to the right, agree=0.626, adj=0.232, (0 split)
## 
## Node number 29: 143 observations,    complexity param=0.02872063
##   predicted class=Y  expected loss=0.3706294  P(node) =0.09178434
##     class counts:    53    90
##    probabilities: 0.371 0.629 
##   left son=58 (61 obs) right son=59 (82 obs)
##   Primary splits:
##       yedge  < 5.5  to the left,  improve=10.253870, (0 missing)
##       xybar  < 11.5 to the right, improve= 8.517798, (0 missing)
##       xy2bar < 4.5  to the left,  improve= 7.608809, (0 missing)
##       ybar   < 6.5  to the right, improve= 7.417780, (0 missing)
##       x2ybar < 5.5  to the left,  improve= 6.410638, (0 missing)
##   Surrogate splits:
##       xbox   < 4.5  to the left,  agree=0.678, adj=0.246, (0 split)
##       ybox   < 9.5  to the left,  agree=0.657, adj=0.197, (0 split)
##       height < 6.5  to the left,  agree=0.643, adj=0.164, (0 split)
##       onpix  < 4.5  to the left,  agree=0.636, adj=0.148, (0 split)
##       width  < 4.5  to the left,  agree=0.629, adj=0.131, (0 split)
## 
## Node number 46: 9 observations
##   predicted class=N  expected loss=0.4444444  P(node) =0.005776637
##     class counts:     5     4
##    probabilities: 0.556 0.444 
## 
## Node number 47: 25 observations
##   predicted class=Y  expected loss=0  P(node) =0.01604621
##     class counts:     0    25
##    probabilities: 0.000 1.000 
## 
## Node number 50: 22 observations
##   predicted class=N  expected loss=0.2272727  P(node) =0.01412067
##     class counts:    17     5
##    probabilities: 0.773 0.227 
## 
## Node number 51: 8 observations
##   predicted class=Y  expected loss=0  P(node) =0.005134788
##     class counts:     0     8
##    probabilities: 0.000 1.000 
## 
## Node number 56: 56 observations
##   predicted class=N  expected loss=0.01785714  P(node) =0.03594352
##     class counts:    55     1
##    probabilities: 0.982 0.018 
## 
## Node number 57: 59 observations,    complexity param=0.01305483
##   predicted class=N  expected loss=0.440678  P(node) =0.03786906
##     class counts:    33    26
##    probabilities: 0.559 0.441 
##   left son=114 (13 obs) right son=115 (46 obs)
##   Primary splits:
##       x2bar     < 3.5  to the left,  improve=6.476050, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=5.626751, (0 missing)
##       y2bar     < 3.5  to the left,  improve=5.084746, (0 missing)
##       yedgexcor < 5.5  to the left,  improve=4.412505, (0 missing)
##       ybox      < 5.5  to the left,  improve=4.124746, (0 missing)
##   Surrogate splits:
##       x2ybar    < 3    to the left,  agree=0.898, adj=0.538, (0 split)
##       ybar      < 5.5  to the left,  agree=0.881, adj=0.462, (0 split)
##       yedgexcor < 5.5  to the left,  agree=0.864, adj=0.385, (0 split)
##       xybar     < 4.5  to the left,  agree=0.831, adj=0.231, (0 split)
##       width     < 8.5  to the right, agree=0.814, adj=0.154, (0 split)
## 
## Node number 58: 61 observations,    complexity param=0.02349869
##   predicted class=N  expected loss=0.4098361  P(node) =0.03915276
##     class counts:    36    25
##    probabilities: 0.590 0.410 
##   left son=116 (24 obs) right son=117 (37 obs)
##   Primary splits:
##       xbar      < 7.5  to the left,  improve=8.436125, (0 missing)
##       ybar      < 6.5  to the right, improve=7.674863, (0 missing)
##       xybar     < 9.5  to the left,  improve=7.231274, (0 missing)
##       yedgexcor < 9.5  to the left,  improve=6.183752, (0 missing)
##       x2bar     < 3.5  to the right, improve=5.090947, (0 missing)
##   Surrogate splits:
##       xybar     < 7.5  to the left,  agree=0.885, adj=0.708, (0 split)
##       x2ybar    < 4.5  to the right, agree=0.770, adj=0.417, (0 split)
##       x2bar     < 4.5  to the right, agree=0.738, adj=0.333, (0 split)
##       yedgexcor < 8.5  to the left,  agree=0.738, adj=0.333, (0 split)
##       width     < 3.5  to the left,  agree=0.656, adj=0.125, (0 split)
## 
## Node number 59: 82 observations,    complexity param=0.005221932
##   predicted class=Y  expected loss=0.2073171  P(node) =0.05263158
##     class counts:    17    65
##    probabilities: 0.207 0.793 
##   left son=118 (12 obs) right son=119 (70 obs)
##   Primary splits:
##       width  < 8.5  to the right, improve=3.975029, (0 missing)
##       xybar  < 10.5 to the right, improve=3.934077, (0 missing)
##       x2ybar < 5.5  to the left,  improve=2.956050, (0 missing)
##       xbox   < 6.5  to the right, improve=2.259490, (0 missing)
##       ybox   < 12.5 to the right, improve=1.951220, (0 missing)
##   Surrogate splits:
##       xbox < 8.5  to the right, agree=0.890, adj=0.250, (0 split)
##       ybox < 13.5 to the right, agree=0.878, adj=0.167, (0 split)
## 
## Node number 114: 13 observations
##   predicted class=N  expected loss=0  P(node) =0.008344031
##     class counts:    13     0
##    probabilities: 1.000 0.000 
## 
## Node number 115: 46 observations,    complexity param=0.01305483
##   predicted class=Y  expected loss=0.4347826  P(node) =0.02952503
##     class counts:    20    26
##    probabilities: 0.435 0.565 
##   left son=230 (29 obs) right son=231 (17 obs)
##   Primary splits:
##       xybar     < 6.5  to the right, improve=7.622894, (0 missing)
##       y2bar     < 3.5  to the left,  improve=5.396574, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=5.157715, (0 missing)
##       ybox      < 7.5  to the left,  improve=4.886473, (0 missing)
##       x2bar     < 5.5  to the right, improve=3.133696, (0 missing)
##   Surrogate splits:
##       xbar   < 8.5  to the left,  agree=0.739, adj=0.294, (0 split)
##       x2ybar < 6.5  to the left,  agree=0.739, adj=0.294, (0 split)
##       x2bar  < 4.5  to the right, agree=0.696, adj=0.176, (0 split)
##       ybox   < 9.5  to the left,  agree=0.652, adj=0.059, (0 split)
##       height < 4.5  to the right, agree=0.652, adj=0.059, (0 split)
## 
## Node number 116: 24 observations
##   predicted class=N  expected loss=0.08333333  P(node) =0.01540436
##     class counts:    22     2
##    probabilities: 0.917 0.083 
## 
## Node number 117: 37 observations,    complexity param=0.01044386
##   predicted class=Y  expected loss=0.3783784  P(node) =0.0237484
##     class counts:    14    23
##    probabilities: 0.378 0.622 
##   left son=234 (14 obs) right son=235 (23 obs)
##   Primary splits:
##       ybar   < 7.5  to the right, improve=3.150747, (0 missing)
##       xbox   < 4.5  to the right, improve=2.200727, (0 missing)
##       x2bar  < 4.5  to the right, improve=1.976834, (0 missing)
##       xy2bar < 5.5  to the left,  improve=1.948263, (0 missing)
##       xybar  < 9.5  to the left,  improve=1.492072, (0 missing)
##   Surrogate splits:
##       xbar      < 9.5  to the left,  agree=0.73, adj=0.286, (0 split)
##       x2bar     < 3.5  to the right, agree=0.73, adj=0.286, (0 split)
##       y2bar     < 5.5  to the left,  agree=0.73, adj=0.286, (0 split)
##       xy2bar    < 6.5  to the left,  agree=0.73, adj=0.286, (0 split)
##       yedgexcor < 8.5  to the left,  agree=0.73, adj=0.286, (0 split)
## 
## Node number 118: 12 observations
##   predicted class=N  expected loss=0.4166667  P(node) =0.007702182
##     class counts:     7     5
##    probabilities: 0.583 0.417 
## 
## Node number 119: 70 observations
##   predicted class=Y  expected loss=0.1428571  P(node) =0.0449294
##     class counts:    10    60
##    probabilities: 0.143 0.857 
## 
## Node number 230: 29 observations,    complexity param=0.01305483
##   predicted class=N  expected loss=0.3448276  P(node) =0.01861361
##     class counts:    19    10
##    probabilities: 0.655 0.345 
##   left son=460 (16 obs) right son=461 (13 obs)
##   Primary splits:
##       y2bar     < 3.5  to the left,  improve=5.689987, (0 missing)
##       ybox      < 7.5  to the left,  improve=5.603448, (0 missing)
##       xedgeycor < 7.5  to the left,  improve=2.285266, (0 missing)
##       x2bar     < 5.5  to the right, improve=1.718833, (0 missing)
##       xedge     < 4.5  to the right, improve=1.303448, (0 missing)
##   Surrogate splits:
##       ybox   < 7.5  to the left,  agree=0.828, adj=0.615, (0 split)
##       xbox   < 5.5  to the left,  agree=0.690, adj=0.308, (0 split)
##       width  < 7.5  to the left,  agree=0.655, adj=0.231, (0 split)
##       x2ybar < 5.5  to the right, agree=0.655, adj=0.231, (0 split)
##       xedge  < 3.5  to the right, agree=0.655, adj=0.231, (0 split)
## 
## Node number 231: 17 observations
##   predicted class=Y  expected loss=0.05882353  P(node) =0.01091142
##     class counts:     1    16
##    probabilities: 0.059 0.941 
## 
## Node number 234: 14 observations
##   predicted class=N  expected loss=0.3571429  P(node) =0.008985879
##     class counts:     9     5
##    probabilities: 0.643 0.357 
## 
## Node number 235: 23 observations
##   predicted class=Y  expected loss=0.2173913  P(node) =0.01476252
##     class counts:     5    18
##    probabilities: 0.217 0.783 
## 
## Node number 460: 16 observations
##   predicted class=N  expected loss=0.0625  P(node) =0.01026958
##     class counts:    15     1
##    probabilities: 0.938 0.062 
## 
## Node number 461: 13 observations
##   predicted class=Y  expected loss=0.3076923  P(node) =0.008344031
##     class counts:     4     9
##    probabilities: 0.308 0.692 
## 
## n= 1558 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 1558 383 N (0.754172015 0.245827985)  
##     2) yedge< 4.5 824  31 N (0.962378641 0.037621359)  
##       4) yedgexcor< 8.5 607   2 N (0.996705107 0.003294893) *
##       5) yedgexcor>=8.5 217  29 N (0.866359447 0.133640553)  
##        10) xedgeycor< 7.5 152   0 N (1.000000000 0.000000000) *
##        11) xedgeycor>=7.5 65  29 N (0.553846154 0.446153846)  
##          22) xedgeycor>=8.5 31   0 N (1.000000000 0.000000000) *
##          23) xedgeycor< 8.5 34   5 Y (0.147058824 0.852941176)  
##            46) ybox>=5.5 9   4 N (0.555555556 0.444444444) *
##            47) ybox< 5.5 25   0 Y (0.000000000 1.000000000) *
##     3) yedge>=4.5 734 352 N (0.520435967 0.479564033)  
##       6) xy2bar>=7.5 284  47 N (0.834507042 0.165492958)  
##        12) yedge< 7.5 244  24 N (0.901639344 0.098360656)  
##          24) xybar< 9.5 214  11 N (0.948598131 0.051401869) *
##          25) xybar>=9.5 30  13 N (0.566666667 0.433333333)  
##            50) yedge< 6.5 22   5 N (0.772727273 0.227272727) *
##            51) yedge>=6.5 8   0 Y (0.000000000 1.000000000) *
##        13) yedge>=7.5 40  17 Y (0.425000000 0.575000000)  
##          26) x2bar>=5.5 13   0 N (1.000000000 0.000000000) *
##          27) x2bar< 5.5 27   4 Y (0.148148148 0.851851852) *
##       7) xy2bar< 7.5 450 145 Y (0.322222222 0.677777778)  
##        14) xedge>=2.5 258 117 N (0.546511628 0.453488372)  
##          28) y2bar< 4.5 115  27 N (0.765217391 0.234782609)  
##            56) xybar>=7.5 56   1 N (0.982142857 0.017857143) *
##            57) xybar< 7.5 59  26 N (0.559322034 0.440677966)  
##             114) x2bar< 3.5 13   0 N (1.000000000 0.000000000) *
##             115) x2bar>=3.5 46  20 Y (0.434782609 0.565217391)  
##               230) xybar>=6.5 29  10 N (0.655172414 0.344827586)  
##                 460) y2bar< 3.5 16   1 N (0.937500000 0.062500000) *
##                 461) y2bar>=3.5 13   4 Y (0.307692308 0.692307692) *
##               231) xybar< 6.5 17   1 Y (0.058823529 0.941176471) *
##          29) y2bar>=4.5 143  53 Y (0.370629371 0.629370629)  
##            58) yedge< 5.5 61  25 N (0.590163934 0.409836066)  
##             116) xbar< 7.5 24   2 N (0.916666667 0.083333333) *
##             117) xbar>=7.5 37  14 Y (0.378378378 0.621621622)  
##               234) ybar>=7.5 14   5 N (0.642857143 0.357142857) *
##               235) ybar< 7.5 23   5 Y (0.217391304 0.782608696) *
##            59) yedge>=5.5 82  17 Y (0.207317073 0.792682927)  
##             118) width>=8.5 12   5 N (0.583333333 0.416666667) *
##             119) width< 8.5 70  10 Y (0.142857143 0.857142857) *
##        15) xedge< 2.5 192   4 Y (0.020833333 0.979166667) *
```

![](Letter_Recognition_B_files/figure-html/fit.models-56.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                              0                           1175
## 2   Y                              0                            383
##           Reference
## Prediction    N    Y
##          N 1109   17
##          Y   66  366
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1109                             66
## 2   Y                             17                            366
##           Reference
## Prediction    N    Y
##          N 1109   17
##          Y   66  366
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1109                             66
## 2   Y                             17                            366
##           Reference
## Prediction    N    Y
##          N 1126   22
##          Y   49  361
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1126                             49
## 2   Y                             22                            361
##           Reference
## Prediction    N    Y
##          N 1135   27
##          Y   40  356
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1135                             40
## 2   Y                             27                            356
##           Reference
## Prediction    N    Y
##          N 1147   36
##          Y   28  347
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1147                             28
## 2   Y                             36                            347
##           Reference
## Prediction    N    Y
##          N 1147   36
##          Y   28  347
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1147                             28
## 2   Y                             36                            347
##           Reference
## Prediction    N    Y
##          N 1151   45
##          Y   24  338
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1151                             24
## 2   Y                             45                            338
##           Reference
## Prediction    N    Y
##          N 1156   63
##          Y   19  320
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1156                             19
## 2   Y                             63                            320
##           Reference
## Prediction    N    Y
##          N 1170  146
##          Y    5  237
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1170                              5
## 2   Y                            146                            237
##           Reference
## Prediction    N    Y
##          N 1175  350
##          Y    0   33
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1175                              0
## 2   Y                            350                             33
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.8981595
## 3        0.2 0.8981595
## 4        0.3 0.9104666
## 5        0.4 0.9139923
## 6        0.5 0.9155673
## 7        0.6 0.9155673
## 8        0.7 0.9073826
## 9        0.8 0.8864266
## 10       0.9 0.7584000
## 11       1.0 0.1586538
```

![](Letter_Recognition_B_files/figure-html/fit.models-57.png) 

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.fit"
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1147                             28
## 2   Y                             36                            347
##           Reference
## Prediction    N    Y
##          N 1147   36
##          Y   28  347
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1147                             28
## 2   Y                             36                            347
##          Prediction
## Reference    N    Y
##         N 1147   28
##         Y   36  347
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.589217e-01   8.884296e-01   9.478442e-01   9.682236e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##  4.501237e-108   3.815739e-01
```

![](Letter_Recognition_B_files/figure-html/fit.models-58.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                              0                           1175
## 2   Y                              0                            383
##           Reference
## Prediction    N    Y
##          N 1091   21
##          Y   84  362
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1091                             84
## 2   Y                             21                            362
##           Reference
## Prediction    N    Y
##          N 1091   21
##          Y   84  362
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1091                             84
## 2   Y                             21                            362
##           Reference
## Prediction    N    Y
##          N 1106   27
##          Y   69  356
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1106                             69
## 2   Y                             27                            356
##           Reference
## Prediction    N    Y
##          N 1114   32
##          Y   61  351
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1114                             61
## 2   Y                             32                            351
##           Reference
## Prediction    N    Y
##          N 1121   51
##          Y   54  332
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1121                             54
## 2   Y                             51                            332
##           Reference
## Prediction    N    Y
##          N 1121   51
##          Y   54  332
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1121                             54
## 2   Y                             51                            332
##           Reference
## Prediction    N    Y
##          N 1125   56
##          Y   50  327
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1125                             50
## 2   Y                             56                            327
##           Reference
## Prediction    N    Y
##          N 1138   63
##          Y   37  320
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1138                             37
## 2   Y                             63                            320
##           Reference
## Prediction    N    Y
##          N 1158  135
##          Y   17  248
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1158                             17
## 2   Y                            135                            248
##           Reference
## Prediction    N    Y
##          N 1174  347
##          Y    1   36
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1174                              1
## 2   Y                            347                             36
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.8733414
## 3        0.2 0.8733414
## 4        0.3 0.8811881
## 5        0.4 0.8830189
## 6        0.5 0.8634590
## 7        0.6 0.8634590
## 8        0.7 0.8605263
## 9        0.8 0.8648649
## 10       0.9 0.7654321
## 11       1.0 0.1714286
```

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1114                             61
## 2   Y                             32                            351
##           Reference
## Prediction    N    Y
##          N 1114   32
##          Y   61  351
##   isB isB.predict.All.X.cp.0.rpart.N isB.predict.All.X.cp.0.rpart.Y
## 1   N                           1114                             61
## 2   Y                             32                            351
##          Prediction
## Reference    N    Y
##         N 1114   61
##         Y   32  351
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.403081e-01   8.430215e-01   9.273697e-01   9.515539e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##   3.509787e-85   3.690585e-03 
##           model_id model_method
## 1 All.X.cp.0.rpart        rpart
##                                                                                                                   feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                       0.57                 0.091
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9843698                    0.6       0.9155673        0.9589217
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9478442             0.9682236     0.8884296   0.9673929
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.8830189        0.9403081
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.9273697             0.9515539     0.8430215
## [1] "iterating over method:rf"
## [1] "fitting model: All.X.rf"
## [1] "    indep_vars: xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm"
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

![](Letter_Recognition_B_files/figure-html/fit.models-59.png) 

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

![](Letter_Recognition_B_files/figure-html/fit.models-60.png) ![](Letter_Recognition_B_files/figure-html/fit.models-61.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           3116   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            2   -none-     character
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
## obsLevels          2   -none-     character
```

![](Letter_Recognition_B_files/figure-html/fit.models-62.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                      0                   1175
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1112    0
##          Y   63  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1112                     63
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1168    0
##          Y    7  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1168                      7
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1174    0
##          Y    1  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1174                      1
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175   15
##          Y    0  368
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                     15                    368
##           Reference
## Prediction    N    Y
##          N 1175  115
##          Y    0  268
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                    115                    268
##           Reference
## Prediction    N    Y
##          N 1175  372
##          Y    0   11
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                    372                     11
##    threshold    f.score
## 1        0.0 0.39464194
## 2        0.1 0.92400483
## 3        0.2 0.99094437
## 4        0.3 0.99869622
## 5        0.4 1.00000000
## 6        0.5 1.00000000
## 7        0.6 1.00000000
## 8        0.7 1.00000000
## 9        0.8 0.98002663
## 10       0.9 0.82334869
## 11       1.0 0.05583756
```

![](Letter_Recognition_B_files/figure-html/fit.models-63.png) 

```
## [1] "Classifier Probability Threshold: 0.7000 to maximize f.score.fit"
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                     NA
## 2   Y                     NA                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y    0  383
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.000000e+00   1.000000e+00   9.976351e-01   1.000000e+00   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##  1.255752e-191            NaN
```

![](Letter_Recognition_B_files/figure-html/fit.models-64.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                      0                   1175
## 2   Y                      0                    383
##           Reference
## Prediction   N   Y
##          N 922   0
##          Y 253 383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                    922                    253
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1063    0
##          Y  112  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1063                    112
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1127    0
##          Y   48  383
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1127                     48
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1153    5
##          Y   22  378
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1153                     22
## 2   Y                      5                    378
##           Reference
## Prediction    N    Y
##          N 1163   14
##          Y   12  369
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1163                     12
## 2   Y                     14                    369
##           Reference
## Prediction    N    Y
##          N 1171   49
##          Y    4  334
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1171                      4
## 2   Y                     49                    334
##           Reference
## Prediction    N    Y
##          N 1173   79
##          Y    2  304
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1173                      2
## 2   Y                     79                    304
##           Reference
## Prediction    N    Y
##          N 1174  132
##          Y    1  251
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1174                      1
## 2   Y                    132                    251
##           Reference
## Prediction    N    Y
##          N 1175  195
##          Y    0  188
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                    195                    188
##           Reference
## Prediction    N    Y
##          N 1175  381
##          Y    0    2
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1175                      0
## 2   Y                    381                      2
##    threshold    f.score
## 1        0.0 0.39464194
## 2        0.1 0.75171737
## 3        0.2 0.87243736
## 4        0.3 0.94103194
## 5        0.4 0.96551724
## 6        0.5 0.96596859
## 7        0.6 0.92649098
## 8        0.7 0.88243832
## 9        0.8 0.79055118
## 10       0.9 0.65849387
## 11       1.0 0.01038961
```

![](Letter_Recognition_B_files/figure-html/fit.models-65.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1163                     12
## 2   Y                     14                    369
##           Reference
## Prediction    N    Y
##          N 1163   14
##          Y   12  369
##   isB isB.predict.All.X.rf.N isB.predict.All.X.rf.Y
## 1   N                   1163                     12
## 2   Y                     14                    369
##          Prediction
## Reference    N    Y
##         N 1163   12
##         Y   14  369
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.833119e-01   9.549143e-01   9.756432e-01   9.890706e-01   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##  5.946825e-148   8.445193e-01 
##   model_id model_method
## 1 All.X.rf           rf
##                                                                                                                           feats
## 1 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      8.455                 1.288
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1           1                    0.7               1        0.9839538
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9976351                     1     0.9563792   0.9984956
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5       0.9659686        0.9833119
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.9756432             0.9890706     0.9549143
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
## 6              Max.cor.Y.glm              glm
## 7    Interact.High.cor.y.glm              glm
## 8              Low.cor.X.glm              glm
## 9                  All.X.glm              glm
## 10               All.X.rpart            rpart
## 11          All.X.cp.0.rpart            rpart
## 12                  All.X.rf               rf
##                                                                                                                            feats
## 1                                                                                                                         .rnorm
## 2                                                                                                                         .rnorm
## 3                                                                                                                          yedge
## 4                                                                                                                          yedge
## 5                                                                                                                          yedge
## 6                                                                                                                          yedge
## 7                                                         yedge, yedge:xbox, yedge:ybox, yedge:xy2bar, yedge:width, yedge:height
## 8                                     yedge, y2bar, x2ybar, yedgexcor, onpix, x2bar, xedge, xbar, xedgeycor, .rnorm, ybar, xybar
## 9  xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
## 10         xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 11         xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 12 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##    max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1                0                      0.346                 0.003
## 2                0                      0.221                 0.001
## 3                0                      0.800                 0.024
## 4                0                      0.672                 0.022
## 5                3                      1.188                 0.023
## 6                1                      0.882                 0.026
## 7                1                      1.194                 0.053
## 8                1                      1.339                 0.087
## 9                1                      1.482                 0.118
## 10               3                      1.557                 0.093
## 11               0                      0.570                 0.091
## 12               3                      8.455                 1.288
##    max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.5000000                    0.5       0.0000000        0.7541720
## 2    0.4980346                    0.2       0.3946419        0.2458280
## 3    0.5000000                    0.5       0.0000000        0.7541720
## 4    0.8463174                    0.3       0.6302596        0.7349166
## 5    0.8337859                    0.4       0.6302596        0.8074391
## 6    0.8636631                    0.2       0.6302596        0.8145040
## 7    0.9169779                    0.4       0.7255689        0.8697026
## 8    0.9663396                    0.4       0.8445006        0.9152809
## 9    0.9840320                    0.5       0.9078947        0.9467356
## 10   0.9058586                    0.6       0.7743733        0.8934502
## 11   0.9843698                    0.6       0.9155673        0.9589217
## 12   1.0000000                    0.7       1.0000000        0.9839538
##    max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.7320038             0.7753782     0.0000000   0.5000000
## 2              0.2246218             0.2679962     0.0000000   0.4961880
## 3              0.7320038             0.7753782     0.0000000   0.5000000
## 4              0.7122496             0.7566947     0.4537937   0.8452319
## 5              0.7122496             0.7566947     0.3893568   0.8302805
## 6              0.7122496             0.7566947     0.4429064   0.8711827
## 7              0.8506195             0.8848207     0.6247090   0.9226987
## 8              0.9065333             0.9339600     0.7687610   0.9694195
## 9              0.9435727             0.9648107     0.8564668   0.9833165
## 10             0.8797895             0.9107407     0.7024631   0.8990245
## 11             0.9478442             0.9682236     0.8884296   0.9673929
## 12             0.9976351             1.0000000     0.9563792   0.9984956
##    opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                     0.5       0.0000000        0.7541720
## 2                     0.2       0.3946419        0.2458280
## 3                     0.5       0.0000000        0.7541720
## 4                     0.3       0.6337900        0.7426187
## 5                     0.4       0.6337900        0.7426187
## 6                     0.3       0.6511111        0.7984596
## 7                     0.5       0.7323944        0.8780488
## 8                     0.4       0.8490099        0.9216945
## 9                     0.5       0.8980645        0.9492940
## 10                    0.6       0.7715054        0.8908858
## 11                    0.4       0.8830189        0.9403081
## 12                    0.5       0.9659686        0.9833119
##    max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1              0.7320038             0.7753782     0.0000000
## 2              0.2246218             0.2679962     0.0000000
## 3              0.7320038             0.7753782     0.0000000
## 4              0.7201446             0.7641747     0.4617023
## 5              0.7201446             0.7641747     0.4617023
## 6              0.7776652             0.8181225     0.5137918
## 7              0.8607636             0.8938943     0.6540602
## 8              0.9072236             0.9345508     0.7963429
## 9              0.9372030             0.9596531     0.8643243
## 10             0.8743405             0.9059406     0.6999182
## 11             0.9273697             0.9515539     0.8430215
## 12             0.9756432             0.9890706     0.9549143
##    max.AccuracySD.fit max.KappaSD.fit min.aic.fit
## 1                  NA              NA          NA
## 2                  NA              NA          NA
## 3                  NA              NA          NA
## 4                  NA              NA          NA
## 5         0.012174548      0.10613540          NA
## 6         0.003099694      0.02354958   1197.5175
## 7         0.003076970      0.02075419    966.5847
## 8         0.006848514      0.01679136    646.2841
## 9         0.015515234      0.04095663    450.8178
## 10        0.006265633      0.02004643          NA
## 11                 NA              NA          NA
## 12                 NA              NA          NA
```

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
## 6              Max.cor.Y.glm              glm
## 7    Interact.High.cor.y.glm              glm
## 8              Low.cor.X.glm              glm
## 9                  All.X.glm              glm
## 10               All.X.rpart            rpart
## 11          All.X.cp.0.rpart            rpart
## 12                  All.X.rf               rf
##                                                                                                                            feats
## 1                                                                                                                         .rnorm
## 2                                                                                                                         .rnorm
## 3                                                                                                                          yedge
## 4                                                                                                                          yedge
## 5                                                                                                                          yedge
## 6                                                                                                                          yedge
## 7                                                         yedge, yedge:xbox, yedge:ybox, yedge:xy2bar, yedge:width, yedge:height
## 8                                     yedge, y2bar, x2ybar, yedgexcor, onpix, x2bar, xedge, xbar, xedgeycor, .rnorm, ybar, xybar
## 9  xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
## 10         xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 11         xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor
## 12 xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybar, xy2bar, xedge, xedgeycor, yedge, yedgexcor, .rnorm
##    max.nTuningRuns max.auc.fit opt.prob.threshold.fit max.f.score.fit
## 1                0   0.5000000                    0.5       0.0000000
## 2                0   0.4980346                    0.2       0.3946419
## 3                0   0.5000000                    0.5       0.0000000
## 4                0   0.8463174                    0.3       0.6302596
## 5                3   0.8337859                    0.4       0.6302596
## 6                1   0.8636631                    0.2       0.6302596
## 7                1   0.9169779                    0.4       0.7255689
## 8                1   0.9663396                    0.4       0.8445006
## 9                1   0.9840320                    0.5       0.9078947
## 10               3   0.9058586                    0.6       0.7743733
## 11               0   0.9843698                    0.6       0.9155673
## 12               3   1.0000000                    0.7       1.0000000
##    max.Accuracy.fit max.Kappa.fit max.auc.OOB opt.prob.threshold.OOB
## 1         0.7541720     0.0000000   0.5000000                    0.5
## 2         0.2458280     0.0000000   0.4961880                    0.2
## 3         0.7541720     0.0000000   0.5000000                    0.5
## 4         0.7349166     0.4537937   0.8452319                    0.3
## 5         0.8074391     0.3893568   0.8302805                    0.4
## 6         0.8145040     0.4429064   0.8711827                    0.3
## 7         0.8697026     0.6247090   0.9226987                    0.5
## 8         0.9152809     0.7687610   0.9694195                    0.4
## 9         0.9467356     0.8564668   0.9833165                    0.5
## 10        0.8934502     0.7024631   0.8990245                    0.6
## 11        0.9589217     0.8884296   0.9673929                    0.4
## 12        0.9839538     0.9563792   0.9984956                    0.5
##    max.f.score.OOB max.Accuracy.OOB max.Kappa.OOB
## 1        0.0000000        0.7541720     0.0000000
## 2        0.3946419        0.2458280     0.0000000
## 3        0.0000000        0.7541720     0.0000000
## 4        0.6337900        0.7426187     0.4617023
## 5        0.6337900        0.7426187     0.4617023
## 6        0.6511111        0.7984596     0.5137918
## 7        0.7323944        0.8780488     0.6540602
## 8        0.8490099        0.9216945     0.7963429
## 9        0.8980645        0.9492940     0.8643243
## 10       0.7715054        0.8908858     0.6999182
## 11       0.8830189        0.9403081     0.8430215
## 12       0.9659686        0.9833119     0.9549143
##    inv.elapsedtime.everything inv.elapsedtime.final  inv.aic.fit
## 1                   2.8901734           333.3333333           NA
## 2                   4.5248869          1000.0000000           NA
## 3                   1.2500000            41.6666667           NA
## 4                   1.4880952            45.4545455           NA
## 5                   0.8417508            43.4782609           NA
## 6                   1.1337868            38.4615385 0.0008350608
## 7                   0.8375209            18.8679245 0.0010345705
## 8                   0.7468260            11.4942529 0.0015473072
## 9                   0.6747638             8.4745763 0.0022181911
## 10                  0.6422608            10.7526882           NA
## 11                  1.7543860            10.9890110           NA
## 12                  0.1182732             0.7763975           NA
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
## 12. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 5 rows containing missing values (geom_path).
```

```
## Warning: Removed 88 rows containing missing values (geom_point).
```

```
## Warning: Removed 8 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

![](Letter_Recognition_B_files/figure-html/fit.models-66.png) 

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

![](Letter_Recognition_B_files/figure-html/fit.models-67.png) 

```r
model_evl_terms <- c(NULL)
for (metric in glb_model_evl_criteria)
    model_evl_terms <- c(model_evl_terms, 
                    ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse=" "))
print(tmp_models_df <- orderBy(model_sel_frmla, glb_models_df)[, c("model_id", glb_model_evl_criteria)])
```

```
##                     model_id max.Accuracy.OOB max.Kappa.OOB min.aic.fit
## 12                  All.X.rf        0.9833119     0.9549143          NA
## 9                  All.X.glm        0.9492940     0.8643243    450.8178
## 11          All.X.cp.0.rpart        0.9403081     0.8430215          NA
## 8              Low.cor.X.glm        0.9216945     0.7963429    646.2841
## 10               All.X.rpart        0.8908858     0.6999182          NA
## 7    Interact.High.cor.y.glm        0.8780488     0.6540602    966.5847
## 6              Max.cor.Y.glm        0.7984596     0.5137918   1197.5175
## 1          MFO.myMFO_classfr        0.7541720     0.0000000          NA
## 3       Max.cor.Y.cv.0.rpart        0.7541720     0.0000000          NA
## 4  Max.cor.Y.cv.0.cp.0.rpart        0.7426187     0.4617023          NA
## 5            Max.cor.Y.rpart        0.7426187     0.4617023          NA
## 2    Random.myrandom_classfr        0.2458280     0.0000000          NA
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
## 12. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 5 rows containing missing values (geom_path).
```

```
## Warning: Removed 22 rows containing missing values (geom_point).
```

```
## Warning: Removed 8 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

![](Letter_Recognition_B_files/figure-html/fit.models-68.png) 

```r
print("Metrics used for model selection:"); print(model_sel_frmla)
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB + min.aic.fit
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

![](Letter_Recognition_B_files/figure-html/fit.models-69.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           3116   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            2   -none-     character
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
## obsLevels          2   -none-     character
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

![](Letter_Recognition_B_files/figure-html/fit.models-70.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8            fit.models                5                0  11.228
## elapsed9 fit.data.training.all                6                0  75.561
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
##             importance        id fit.feat
## yedge     100.00000000     yedge     TRUE
## xedgeycor  75.46759209 xedgeycor     TRUE
## y2bar      70.36052015     y2bar     TRUE
## xy2bar     56.81268057    xy2bar     TRUE
## yedgexcor  37.19555937 yedgexcor     TRUE
## x2ybar     33.19388395    x2ybar     TRUE
## ybar       31.05508754      ybar     TRUE
## xedge      27.74346173     xedge     TRUE
## xybar      24.71489344     xybar     TRUE
## x2bar      14.21742517     x2bar     TRUE
## xbar       13.15738565      xbar     TRUE
## onpix       8.27549131     onpix     TRUE
## width       2.42219827     width     TRUE
## ybox        1.78944990      ybox     TRUE
## height      1.08977988    height     TRUE
## xbox        0.05863901      xbox     TRUE
## .rnorm      0.00000000    .rnorm     TRUE
## [1] "fitting model: Final.rf"
## [1] "    indep_vars: yedge, xedgeycor, y2bar, xy2bar, yedgexcor, x2ybar, ybar, xedge, xybar, x2bar, xbar, onpix, width, ybox, height, xbox, .rnorm"
## + : mtry= 2 
## - : mtry= 2 
## + : mtry= 9 
## - : mtry= 9 
## + : mtry=17 
## - : mtry=17 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 9 on full training set
```

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_0-1.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_0-2.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1558   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           3116   matrix     numeric  
## oob.times       1558   -none-     numeric  
## classes            2   -none-     character
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
## obsLevels          2   -none-     character
```

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_0-3.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 1175  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                      0                   1175
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1128    0
##          Y   47  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1128                     47
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1168    0
##          Y    7  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1168                      7
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1173    0
##          Y    2  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1173                      2
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##           Reference
## Prediction    N    Y
##          N 1175   11
##          Y    0  372
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                     11                    372
##           Reference
## Prediction    N    Y
##          N 1175   68
##          Y    0  315
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                     68                    315
##           Reference
## Prediction    N    Y
##          N 1175  283
##          Y    0  100
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                    283                    100
##    threshold   f.score
## 1        0.0 0.3946419
## 2        0.1 0.9421894
## 3        0.2 0.9909444
## 4        0.3 0.9973958
## 5        0.4 1.0000000
## 6        0.5 1.0000000
## 7        0.6 1.0000000
## 8        0.7 1.0000000
## 9        0.8 0.9854305
## 10       0.9 0.9025788
## 11       1.0 0.4140787
```

```
## [1] "Classifier Probability Threshold: 0.7000 to maximize f.score.fit"
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                     NA
## 2   Y                     NA                    383
##           Reference
## Prediction    N    Y
##          N 1175    0
##          Y    0  383
##   isB isB.predict.Final.rf.N isB.predict.Final.rf.Y
## 1   N                   1175                      0
## 2   Y                      0                    383
##          Prediction
## Reference    N    Y
##         N 1175    0
##         Y    0  383
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.000000e+00   1.000000e+00   9.976351e-01   1.000000e+00   7.541720e-01 
## AccuracyPValue  McnemarPValue 
##  1.255752e-191            NaN
```

```
## Warning in mypredict_mdl(mdl, df = fit_df, rsp_var, rsp_var_out,
## model_id_method, : Expecting 1 metric: Accuracy; recd: Accuracy, Kappa;
## retaining Accuracy only
```

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_0-4.png) 

```
##   model_id model_method
## 1 Final.rf           rf
##                                                                                                                           feats
## 1 yedge, xedgeycor, y2bar, xy2bar, yedgexcor, x2ybar, ybar, xedge, xybar, x2bar, xbar, onpix, width, ybox, height, xbox, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      7.044                 1.808
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1           1                    0.7               1        0.9839538
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.9976351                     1     0.9565334
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
## elapsed9  fit.data.training.all                6                0  75.561
## elapsed10 fit.data.training.all                6                1  92.598
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
##           id        cor.y exclude.as.feat   cor.y.abs cor.low  importance
## 16     yedge  0.561065858               0 0.561065858       1 100.0000000
## 13     y2bar  0.517624377               0 0.517624377       1  57.0826864
## 11    xy2bar -0.013432353               0 0.013432353       0  42.8197320
## 10 xedgeycor  0.059957233               0 0.059957233       1  40.7056910
## 9      xedge  0.077198024               0 0.077198024       1  30.4551112
## 12     xybar -0.146207921               0 0.146207921       1  15.9542002
## 17 yedgexcor  0.219787666               0 0.219787666       1  11.2683878
## 14      ybar -0.026531619               0 0.026531619       1  10.9681342
## 5      x2bar  0.147216267               0 0.147216267       1   9.5116052
## 6     x2ybar  0.286395882               0 0.286395882       1   9.2195995
## 8       xbox  0.017494730               0 0.017494730       0   2.6825434
## 7       xbar  0.077078657               0 0.077078657       1   2.1490493
## 4      width -0.023742242               0 0.023742242       0   2.1053619
## 2     height -0.027185778               0 0.027185778       0   0.7559765
## 15      ybox -0.010644069               0 0.010644069       0   0.3430926
## 3      onpix  0.216051154               0 0.216051154       1   0.2317614
## 1     .rnorm -0.007715895               0 0.007715895       1   0.0000000
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

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-1.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-2.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-3.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-4.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-5.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-6.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-7.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-8.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-9.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-10.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-11.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-12.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-13.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-14.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-15.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-16.png) ![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-17.png) 

```
##     letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 22       A    3    3     5      4     1    7    6     3     0     7      0
## 278      B    5    4     5      6     4    7    9     9     8     7      5
## 951      B    5   11     7      8    11    8    9     5     3     6      7
##     xy2bar xedge xedgeycor yedge yedgexcor     .rnorm isB
## 22       8     2         7     1         8 -1.3010367   N
## 278      7     2         8     9         9 -0.9151249   Y
## 951      7     7        11    12         9 -0.0459042   Y
##     isB.predict.Final.rf.prob isB.predict.Final.rf
## 22                      0.000                    N
## 278                     0.994                    Y
## 951                     0.924                    Y
##     isB.predict.Final.rf.accurate .label
## 22                           TRUE    .22
## 278                          TRUE   .278
## 951                          TRUE   .951
```

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-18.png) 

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

![](Letter_Recognition_B_files/figure-html/fit.data.training.all_1-19.png) 

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
## elapsed10 fit.data.training.all                6                1  92.598
## elapsed11      predict.data.new                7                0 105.697
```

## Step `7`: predict data.new

```r
# Compute final model predictions
glb_newent_df <- glb_get_predictions(glb_newent_df)
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

![](Letter_Recognition_B_files/figure-html/predict.data.new-1.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-2.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-3.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-4.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-5.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-6.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-7.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-8.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-9.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-10.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-11.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-12.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-13.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-14.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-15.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-16.png) ![](Letter_Recognition_B_files/figure-html/predict.data.new-17.png) 

```
##     letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
## 26       B    5    5     5      8     4    7    9    10     8     7      5
## 154      P    4    8     6     12    10    8    9     5     0     8      6
## 496      A    1    3     2      1     1    9    3     2     1     8      2
## 978      B    6   10     9      8    12    8    8     5     3     6      7
##     xy2bar xedge xedgeycor yedge yedgexcor     .rnorm isB
## 26       7     2         8     9        10  0.1320093   Y
## 154      7     5        11     6         6  0.2035100   N
## 496      9     1         6     0         8 -0.2565339   N
## 978      7     7        10    11        10  0.7776949   Y
##     isB.predict.Final.rf.prob isB.predict.Final.rf
## 26                      0.992                    Y
## 154                     0.056                    N
## 496                     0.000                    N
## 978                     0.940                    Y
##     isB.predict.Final.rf.accurate .label
## 26                           TRUE    .26
## 154                          TRUE   .154
## 496                          TRUE   .496
## 978                          TRUE   .978
```

![](Letter_Recognition_B_files/figure-html/predict.data.new-18.png) 

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

![](Letter_Recognition_B_files/figure-html/predict.data.new-19.png) 

```r
print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

![](Letter_Recognition_B_files/figure-html/predict.data.new-20.png) 

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
## 10      fit.data.training.all                6                0  75.561
## 11      fit.data.training.all                6                1  92.598
## 12           predict.data.new                7                0 105.697
## 9                  fit.models                5                0  11.228
## 6            extract_features                3                0   5.170
## 7             select_features                4                0   6.789
## 4         manage_missing_data                2                2   1.600
## 2                cleanse_data                2                0   0.561
## 5          encode_retype_data                2                3   2.043
## 8  remove_correlated_features                4                1   6.993
## 3       inspectORexplore.data                2                1   0.595
## 1                 import_data                1                0   0.003
##    elapsed_diff
## 10       64.333
## 11       17.037
## 12       13.099
## 9         4.235
## 6         3.127
## 7         1.619
## 4         1.005
## 2         0.558
## 5         0.443
## 8         0.204
## 3         0.034
## 1         0.000
```

```
## [1] "Total Elapsed Time: 105.697 secs"
```

![](Letter_Recognition_B_files/figure-html/print_sessionInfo-1.png) 

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
##  [4] ROCR_1.0-7          gplots_2.16.0       caret_6.0-41       
##  [7] lattice_0.20-31     sqldf_0.4-10        RSQLite_1.0.0      
## [10] DBI_0.3.1           gsubfn_0.6-6        proto_0.3-10       
## [13] reshape2_1.4.1      plyr_1.8.1          caTools_1.17.1     
## [16] doBy_4.5-13         survival_2.38-1     ggplot2_1.0.1      
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gdata_2.13.3       
## [16] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.6    
## [19] iterators_1.0.7     KernSmooth_2.23-14  knitr_1.9          
## [22] labeling_0.3        lme4_1.1-7          MASS_7.3-40        
## [25] Matrix_1.2-0        mgcv_1.8-6          minqa_1.2.4        
## [28] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [31] nnet_7.3-9          parallel_3.1.3      pbkrtest_0.4-2     
## [34] quantreg_5.11       RColorBrewer_1.1-2  Rcpp_0.11.5        
## [37] rmarkdown_0.5.1     scales_0.2.4        SparseM_1.6        
## [40] splines_3.1.3       stringr_0.6.2       tools_3.1.3        
## [43] yaml_2.1.13
```
