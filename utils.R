require(dplyr)
require(tidyr)
require(RSNNS)

create_folds <- function(x, k=5){
  k = 5
  set.seed(2020)
  shuffle <- sample(length(x), length(unique(x)), replace=FALSE)
  result <- split(shuffle, cut(seq_along(shuffle), k, labels = FALSE)) 
  result
}

wrangle_pbp <- function(pbp){
  #' Performs basic wrangling tasks on a raw pbp dataset. Basic feature engineering here.
  #' @param pbp : df. A dataframe of raw play-by-play data
  #' @return df : wrangled play-by-play data
  
  pbp_wr <- pbp %>%
    filter(qtr <= 4, 
           play == 1) %>%
    mutate(half_min_remaining = half_seconds_remaining/60,
           game_min_remaining = game_seconds_remaining/60,
           game_min_remaining_inv = ifelse(game_min_remaining==0,
                                           1/(1/60),
                                           1/game_min_remaining), 
           log_game_min_remaining_inv = log(game_min_remaining_inv),
           is_home_team = ifelse(posteam == home_team, 1, 0),
           # dummies
           q1 = ifelse(qtr==1, 1, 0),
           q2 = ifelse(qtr==2, 1, 0),
           q3 = ifelse(qtr==3, 1, 0),
           q4 = ifelse(qtr==4, 1, 0),
           d1 = ifelse(down==1, 1, 0),
           d2 = ifelse(down==2, 1, 0),
           d3 = ifelse(down==3, 1, 0),
           d4 = ifelse(down==4, 1, 0),
           # how much time could you burn kneeling and the other team using TOs
           pos_kneel_burn_sec = 39 * (4 - down - defteam_timeouts_remaining) * 
              (4 - down - defteam_timeouts_remaining > 0) + 5, # assume you can burn 5 seconds by punting on 4th
           # can the offensive team run out the clock
           pos_can_kneel = ifelse(pos_kneel_burn_sec >= game_seconds_remaining, 1, 0),
           # the same, but for the defense on immediate turnover
           def_kneel_burn_sec = 39 * (3 - posteam_timeouts_remaining) + 5,
           # can the defensive team run out the clock if they get thte ball
           def_can_kneel = ifelse(def_kneel_burn_sec >= game_seconds_remaining, 1, 0),
           # effective time left in game, depending on possession
           pos_min_remaining = game_min_remaining - pos_kneel_burn_sec/60,
           def_min_remaining = game_min_remaining - def_kneel_burn_sec/60,
           game_is_over = ifelse(score_differential > 0 & pos_can_kneel, 1, 0),
           # timeouts, split out by half
           posteam_timeouts_remaining_total = ifelse(game_min_remaining > 30, posteam_timeouts_remaining + 3, posteam_timeouts_remaining),
           defteam_timeouts_remaining_total = ifelse(game_min_remaining > 30, defteam_timeouts_remaining + 3, defteam_timeouts_remaining)
        )
  
  game_results = pbp %>%
    filter(desc == 'END GAME') %>%
    select(game_id, home_score, away_score, home_team, away_team) %>%
    mutate(is_home_win = ifelse(home_score > away_score, 1, 0),
           is_away_win = ifelse(home_score < away_score, 1, 0)) %>%
    select(game_id, is_home_win, is_away_win)
  
  pbp_wr = inner_join(pbp_wr, game_results, by = c("game_id")) %>%
    mutate(
      y = ifelse(
        (posteam == home_team & is_home_win == 1)|(posteam == away_team & is_away_win == 1),
        1, 0),
      y_str = ifelse(y, "Y", "N"),
      y_bool = ifelse(y, T, F)
    ) %>%
    drop_na(all_of(c(CONT_COLNAMES, CAT_COLNAMES)))
  pbp_wr
}

process_data <- function(df, 
                         xvars_cont=CONT_COLNAMES,
                         xvars_cat=CAT_COLNAMES,
                         fit=TRUE,
                         processor=NULL){
  #' Performs preprocessing -- either fits scale or applies fitted scale -- prior to modeling
  #' @param df : data.frame. A dataframe containing data to model
  #' @param xvars_cont : list<str>. A list of continuous features in df to include in model
  #' @param xvars_cat : list<str>. A list of categorical features in df to include in model
  #' @param fit : bool. Whether or not to fit the scale, or apply the scale provided in processor
  #' @param processor : list. A list, as outputted by `process_data()``, with already-fit scales. 
  
  if (fit){
    df_sc <- df
    scaler_stddev <- preProcess(df[xvars_cont], method=c("center", "scale"))
    scaler_minmax <- preProcess(predict(scaler_stddev, df[xvars_cont]), method='range')
    df_sc[xvars_cont] <- predict(scaler_minmax, predict(scaler_stddev, df[xvars_cont]))
    result = list(df=df,
                  scaler_stddev=scaler_stddev,
                  scaler_minmax=scaler_minmax,
                  df_sc=df_sc,
                  xvars_cont=xvars_cont,
                  xvars_cat=xvars_cat)
    
  }else{
    df_sc <- df
    # unpack
    scaler_stddev = processor$scaler_stddev
    scaler_minmax = processor$scaler_minmax
    # scale
    df_sc[processor$xvars_cont] = predict(scaler_minmax, predict(scaler_stddev, df[processor$xvars_cont]))
    result = df_sc
  }
  
  result
}


upsample <- function(df, 
                     ratio=.2, 
                     .FUN=function(x){df %>% filter(abs(score_differential) <= 10, game_min_remaining <= 3)}){
  #' TODO: docstring this
  set.seed(2020)
  samp_df = .FUN(df)
  samp_df = samp_df[sample(1:nrow(samp_df), round(ratio * nrow(df)), replace=T), ]
  bind_rows(df, samp_df)
}

# train_rf <- function(df_train, xvars_cont=CONT_COLNAMES, xvars_cat=CAT_COLNAMES, ...){
#   
#   require(randomForest)
#   df_train$y_str = as.factor(df_train$y_str)
#   
#   clf <- randomForest(
#     formula = y_str ~ .,
#     data = df_train[, c("y_str", xvars_cont, xvars_cat)],
#     # x = df_train[, c(xvars_cont, xvars_cat)],
#     # y = as.factor(df_train$y),
#     ...
#   )
#   
#   clf
# }


train_mlp <- function(df_train, xvars_cont=CONT_COLNAMES, xvars_cat=CAT_COLNAMES, 
                      n_models=5, boot_size=100000, ...){
  #' Preprocesses data and trains multiple MLPs, which are averaged to generate prediction
  #' @param df_train : data.frame. A df containing training data
  #' @param xvars_cont : list<str>. A list of continuous features in df to include in model
  #' @param xvars_cat : list<str>. A list of categorical features in df to include in model
  #' @param n_models : int. Number of models to average over
  #' @param boot_size : Bootstrap size per model. 
  #' @return list : A list containing:
  #'                  * a list of the models
  #'                  * `processor`, the output of process_data()
  
  processor = process_data(df_train, 
                           xvars_cont=xvars_cont,
                           xvars_cat=xvars_cat,
                           fit=TRUE)
  
  clf <- lapply(1:n_models, function(n){
    cat('\nModel: ', n, '/', n_models);
    set.seed(n);
    slc = sample(1:nrow(processor$df_sc), boot_size, replace=T);
    # print(df_train[slc, ] %>% pull(pos_can_kneel) %>% mean());
    # print(df_train[slc, ] %>% filter(pos_can_kneel ==1, score_differential > 0) %>% nrow());
    mlp(
      x = processor$df_sc[slc, c(xvars_cont, xvars_cat)],
      y = processor$df_sc[slc, ] %>% pull(y),
      linOut=F,
      ...
    )
  })
  
  list(clf, processor)
}

predict_mlp <- function(df_val, processor, clf, validate=NULL){
  
  df_val_sc <- process_data(df_val, fit=F, processor=processor)
  pi = lapply(clf, predict, df_val_sc[, c(processor$xvars_cont, processor$xvars_cat)]) %>%
    do.call("cbind", .) %>%
    rowMeans()
  if (validate=='log_loss'){
    log_loss = LogLoss(pi, df_val_sc$y)
    return(log_loss)
  }else if (validate=='acc'){
    acc = Accuracy(round(pi), df_val_sc$y)
    return(acc)
  }else if (validate=='matthews'){
    mw = mltools::mcc(round(pi), as.numeric(df_val_sc$y))
    return(mw)
  }else{
    df_val$pi = pi
    return(df_val)
  }
}


evaluate_gamestate <- function(xvar_dict, clf, processor=NULL, ...){
  #' Performs basic wrangling on provided game state, and then predicts 
  #' from that game state. 
  #' @param xvar_dict: list. A list containing the following key/value pairs:
  #'                         * down : int
  #'                         * ydstogo : int
  #'                         * game_min_remaining : numeric
  #'                         * posteam_timeouts_remaining : int
  #'                         * defteam_timeouts_remaining : int
  #'                         * posteam_score : int
  #'                         * defteam_score : int
  #'                         * yardline_100 : float
  #' @param clf: list or mlp. The models (either a list or a standalone) used to predict
  #' @param processor : list. a processor list, as outputted by process_data(). NULL --> no preprocessing
  #' @return numeric : P(W|gamestate)
  
  df_tidy = data.frame(xvar_dict)
  # add implicitly-defined variables
  xvar_dict$d1 = ifelse(xvar_dict$down==1, 1, 0)
  xvar_dict$d2 = ifelse(xvar_dict$down==2, 1, 0)
  xvar_dict$d3 = ifelse(xvar_dict$down==3, 1, 0)
  xvar_dict$d4 = ifelse(xvar_dict$down==4, 1, 0)
  xvar_dict$score_differential = xvar_dict$posteam_score - xvar_dict$defteam_score
  xvar_dict$game_min_remaining_inv = 1/xvar_dict$game_min_remaining
  # xvar_dict$log_game_min_remaining_inv = log(1/xvar_dict$game_min_remaining)
  
  # adjust timeouts
  xvar_dict$posteam_timeouts_remaining_total = ifelse(xvar_dict$game_min_remaining > 30,
                                                      xvar_dict$posteam_timeouts_remaining + 3, 
                                                      xvar_dict$posteam_timeouts_remaining)
  xvar_dict$defteam_timeouts_remaining_total = ifelse(xvar_dict$game_min_remaining > 30,
                                                      xvar_dict$defteam_timeouts_remaining + 3,
                                                      xvar_dict$defteam_timeouts_remaining)
  
  xhat_unscaled = data.frame(xvar_dict) %>%
    mutate(
      # remake
      game_seconds_remaining = game_min_remaining * 60, 
      # how much time could you burn kneeling and the other team using TOs
      pos_kneel_burn_sec = 39 * (4 - down - defteam_timeouts_remaining) * 
        (4 - down - defteam_timeouts_remaining > 0) + 5, # assume you can burn 5 seconds by punting on 4th
      # can the offensive team run out the clock
      pos_can_kneel = ifelse(pos_kneel_burn_sec >= game_seconds_remaining, 1, 0),
      # the same, but for the defense on immediate turnover
      def_kneel_burn_sec = 39 * (3 - posteam_timeouts_remaining) + 5,
      # can the defensive team run out the clock if they get thte ball
      def_can_kneel = ifelse(def_kneel_burn_sec >= game_seconds_remaining, 1, 0),
      # effective time left in game, depending on possession
      pos_min_remaining = game_min_remaining - pos_kneel_burn_sec/60,
      def_min_remaining = game_min_remaining - def_kneel_burn_sec/60,
      game_is_over = ifelse(score_differential > 0 & pos_can_kneel, 1, 0)
      )
  
  if (!is.null(processor)){
    xhat = process_data(xhat_unscaled, fit=F, processor=processor)
    xhat = xhat[, c(processor$xvars_cont, processor$xvars_cat)]
  }else{
    xhat = xhat_unscaled
  }
  # if (xhat$game_is_over == 1){return(1)}
  if (class(clf) == 'list'){
    pi = lapply(clf, function(x) predict(x, xhat, ...)) %>%
      do.call('rbind', .) %>%
      colMeans()
  }else{
    pi = predict(clf, xhat, ...)
  }
  pi
}

evaluate_dog <- function(xvar_dict, clf, processor=NULL, ...){
  #' Evaluates whether or not to take a delay of game in a particular situation. 
  #' @param xvar_dict: list. A list containing the following key/value pairs:
  #'                         * down : int
  #'                         * ydstogo : int
  #'                         * game_min_remaining : numeric
  #'                         * posteam_timeouts_remaining : int
  #'                         * defteam_timeouts_remaining : int
  #'                         * posteam_score : int
  #'                         * defteam_score : int
  #'                         * yardline_100 : float
  #' @param clf: list or mlp. The models (either a list or a standalone) used to predict
  #' @param processor : list. a processor list, as outputted by process_data(). NULL --> no preprocessing
  #' @return list. A list containing:
  #'               * P(W | gamestate, TO)
  #'               * P(W | gamestate, D.O.G.)
  
  xvar_dict_timeout = xvar_dict; xvar_dict_dog = xvar_dict
  
  if (xvar_dict$posteam_timeouts_remaining){
    xvar_dict_timeout$posteam_timeouts_remaining = xvar_dict_timeout$posteam_timeouts_remaining - 1
  }else{
    warning("No timeouts in your possession!")
  }
  
  xvar_dict_dog$yardline_100 = min(xvar_dict_dog$yardline_100 + 5, 99)
  xvar_dict_dog$ydstogo = xvar_dict_dog$ydstogo + 5
  
  wp_timeout = evaluate_gamestate(xvar_dict_timeout, clf, processor, ...)
  wp_dog = evaluate_gamestate(xvar_dict_dog, clf, processor, ...)
  
  list(wp_timeout, wp_dog)
}


load_mlp <- function(prefix='model', n=10){
  #' Loads models and scalers for an MLP fit
  #' @param prefix : str. prefix of model names
  #' @param n : int. Number of models in ensemble
  #' @return list. List containing: list of models, a `process_data()` output
  model_list = list()
  for (i in 1:n){
    model_list[[i]] = readRDS(paste0(prefix, '-', n, '.RDS'))
  }
  scales = readRDS('processor-fit.RDS')
  return(list(model_list, scales))
}
