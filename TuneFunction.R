library(mlbench)
library(roxygen2)
library(rsample)
library(parsnip)
library(tidymodels)

tune_model = function(
    dataset, outcome_var, algorithm, task, metric_function, tuning_type, 
    tuning_grid = NULL, initial_input = NULL, bayes_iter = NULL
  ) {
  set.seed(29)
  
  # Create model formula
  x_vars = names(dataset)[names(dataset) != `outcome_var`]
  model_formula = reformulate(termlabels = x_vars, response = outcome_var)
  
  # Create train-test split
  splits = initial_split(dataset, prop = 0.7)
  train_data = training(splits)
  test_data = testing(splits)
  
  folds <- vfold_cv(train_data, v = 5)
  
  # preprocessor
  data_recipe = recipe(model_formula, data = train_data) |>
    step_normalize(all_numeric_predictors())
  
  # Train algorithm-specific model
  if(algorithm == "random_forest") {
    # Fit Random Forest model
    model = rand_forest(
      mode = task,
      mtry = tune(),
      min_n = tune(),
      trees = tune()
    )
    
    # define workflow
    model_wf = workflow() |>
      add_model(model) |>
      add_recipe(data_recipe)
    
    # define params
    param_set = model_wf |>
      extract_parameter_set_dials() |>
      update(
        mtry = finalize(mtry(), train_data %>% select(-!!as.name(outcome_var)))
      )
    
  } else if (algorithm == "LASSO") {
    # Fit LASSO model
    if (task == "regression") {
      model = linear_reg(penalty = tune(), mixture = 1.0) |>
        set_engine("glmnet")
    } else {
      model = logistic_reg(penalty = tune(), mixture = 1.0) |>
        set_engine("glmnet")
    }
    
    model_wf = workflow() |>
      add_model(model) |>
      add_recipe(data_recipe)
    
    param_set = model_wf |>
      extract_parameter_set_dials() |>
      update(
        penalty = finalize(penalty(c(0.0, 1.0), trans = NULL))
      )
    
  } else if (algorithm == "ridge") {
    #Fit Ridge model
    if (task == "regression") {
      model = linear_reg(penalty = tune(), mixture = 0.0) |>
        set_engine("glmnet")
    } else {
      model = logistic_reg(penalty = tune(), mixture = 0.0) |>
        set_engine("glmnet")
    }
    
    model_wf = workflow() |>
      add_model(model) |>
      add_recipe(data_recipe)
    
    param_set = model_wf |>
      extract_parameter_set_dials() |>
      update(
        penalty = finalize(penalty(c(0.0, 1.0), trans = NULL))
      )
    
  } else {
    stop(paste0("Algorithm '", algorithm, "' is not currently supported."))
  }
  
  if (tuning_type == "bayes") {
    tune_results = model_wf |>
      tune_bayes(
        iter = bayes_iter,
        resamples = folds,
        param_info = param_set,
        # issue is here!
        metrics = metric_set(!!sym(metric_function)),
        initial = initial_input,
        control = control_bayes(verbose = FALSE)
      )
    estimates = collect_metrics(tune_results) |>
      arrange(.iter)
    best = show_best(tune_results)
  } else {
    tune_results = model_wf |>
      tune_grid(
        resamples = folds,
        grid = tuning_grid,
        metrics = metric_set(!!sym(metric_function)),
      )
    estimates = collect_metrics(tune_results)
    best = show_best(tune_results)
  }
  
  return(list(estimates, best))
}


## TESTS

# data(Shuttle)
# data(iris)

# tune_model(
#   iris[,1:4],
#   "Petal.Width",
#   "ridge",
#   "regression",
#   "rmse",
#   tuning_type = "grid",
#   tuning_grid = grid_regular(
#     penalty(c(0, 10), trans = NULL),
#     levels = 21
#   )
# )
# tune_model(
#   iris[,1:4],
#   "Petal.Width",
#   "random_forest",
#   "regression",
#   "rmse",
#   tuning_type = "bayes",
#   initial_input = 5,
#   bayes_iter = 10
# )
