# fits 10 bootstrapped/upsampled MLPs. 

library(purrr)
library(dplyr)
library(tidyr)
library(glue)
library(lme4)
library(splines)
library(caret)
library(MLmetrics)
library(mltools)
library(ggplot2)
source("utils.R")

# continuous features
CONT_COLNAMES = c("ydstogo", 
                  # "posteam_score", 
                  # "defteam_score", 
                  "score_differential", 
                  "game_min_remaining", 
                  "yardline_100",
                  "posteam_timeouts_remaining_total",
                  "defteam_timeouts_remaining_total",
                  "pos_min_remaining",
                  "def_min_remaining"
)
# discrete features
CAT_COLNAMES = c(paste0("d", 1:4), "pos_can_kneel", "def_can_kneel", "game_is_over")

# seasons available
SEASONS <- 2005:2020

# tune (dev = 2015-17) or nah?
TUNE=F

# based on how tuning/training went
MODEL_PARAMS = list(
  learnFuncParams=c(0.01), 
  size=c(25, 25, 12),
  maxit=750,
  n_models=10
)

main <- function(tune=TUNE, seasons=SEASONS){
  
  # get the data
  pbp <- purrr::map_df(seasons, function(x) {
    readRDS(
      url(
        glue::glue("https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data/play_by_play_{x}.rds")
      )
    )
  })
  # wrangling -- see utils.R
  pbp_wr = wrangle_pbp(pbp)
  
  if (tune){
    # partition tuning and dev set
    df_tune_unscaled = pbp_wr %>% 
      filter(season <= 2014) %>% 
      upsample(., ratio=.2, .FUN=function(x){df %>% filter(abs(score_differential) <= 10, game_min_remaining <= 3)})
    df_val_unscaled = pbp_wr %>% filter(season %in% c(2015, 2016, 2017))
    # Model Tuning ------------------------------------------------------------
    tune_grid <- expand.grid(
      size = list(c(10, 10),
                  c(25, 25),
                  c(37, 37),
                  c(50, 50),
                  c(10, 10, 10),
                  c(25, 25, 20)),
      learnFuncParams = c(.01, .001),
      maxit = c(500, 750, 1000),
      n_models = 5
    )
    tune_grid[, c("log_loss", "acc")] = NA
    
    for (row in 1:nrow(tune_grid)){
      set.seed(2020 + row)
      tune_iter <- train_mlp(df_tune_unscaled,
                             size=tune_grid$size[[row]], 
                             learnFuncParams=tune_grid$learnFuncParams[row], 
                             maxit=tune_grid$maxit[row], 
                             n_models=tune_grid$n_models[row])
      tune_grid$log_loss[row] <- predict_mlp(df_val_unscaled, 
                                             processor=tune_iter[[2]], 
                                             clf=tune_iter[[1]], 
                                             validate='log_loss')
      tune_grid$acc[row] <- predict_mlp(df_val_unscaled, tune_iter[[2]], tune_iter[[1]], 
                                        validate='acc')
    }
    
    # Model Testing -----------------------------------------------------------
    metric = 'log_loss'; fun=min
    best_tune = tune_grid[which(tune_grid[, metric]==fun(tune_grid[, metric])), ]
    yhat_list = list()
    
    # predicting 2018, 2019, 2020
    for (seas in c(2018, 2019, 2020)){
      df_train_unscaled = pbp_wr %>% 
        filter(season <= seas) %>% 
        upsample(., ratio=.2, .FUN=function(x){df %>% filter(abs(score_differential) <= 10, game_min_remaining <= 3)})
      df_test_unscaled = pbp_wr %>% filter(season == seas + 1)
      model_info = train_mlp(df_train_unscaled, 
                             size=best_tune$size[[1]], 
                             learnFuncParams=best_tune$learnFuncParams, 
                             maxit=best_tune$maxit, 
                             n_models=best_tune$n_models)
      yhat_list[[seas]] = predict_slp(df_test_unscaled, model_info[[2]], model_info[[1]], validate='')
    }
    yhat = do.call("rbind", yhat_list)
  }
  
  # Model Fit ---------------------------------------------------------------
  # upsample a few late/close games
  pbp_wr_us = upsample(pbp_wr, ratio=.2, .FUN=function(x){x %>% filter(abs(score_differential) <= 10, game_min_remaining <= 3)})
  model_info = train_mlp(pbp_wr_us, 
                         xvars_cont = CONT_COLNAMES,
                         xvars_cat = CAT_COLNAMES,
                         learnFuncParams=MODEL_PARAMS$learnFuncParams, 
                         size=MODEL_PARAMS$size,
                         maxit=MODEL_PARAMS$maxit,
                         n_models=MODEL_PARAMS$n_models,
                         boot_size=nrow(pbp_wr_us))
  
  for (mm in 1:length(model_info[[1]])){
    saveRDS(model_info[[1]][[mm]], paste0('model-', mm, ".RDS"))
  }
  saveRDS(model_info[[2]][c("scaler_stddev", "scaler_minmax", "xvars_cat", "xvars_cont")],
          'processor-fit.RDS')
  
}

# xvar_dict = list(
#   down=1,
#   ydstogo=20,
#   yardline_100=25,
#   posteam_score=7,
#   defteam_score=10,
#   game_min_remaining=13,
#   posteam_timeouts_remaining=3,
#   defteam_timeouts_remaining=3
# )
# 
# comparison = evaluate_dog(xvar_dict, model_info[[1]], model_info[[2]])
# 
# cat('* Approx. Win Probability w/ Timeout\n\t', comparison[[1]],
#     '\n* Approx. Win Probability w/ D.O.G. \n\t', comparison[[2]])
