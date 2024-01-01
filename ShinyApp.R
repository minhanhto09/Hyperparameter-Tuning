library(future)
library(glue)
library(shinycssloaders)
library(shiny)
library(bslib)
library(plotly)
library(shinythemes)
library(tidyverse)
library(tidymodels)
library(shinyjs)

source("TuneFunction.R")

# Load and subset shuttle data
data(Shuttle)
shuttle_filtered = Shuttle |> 
  filter(Class %in% c("Rad.Flow", "High")) |> 
  select(V1, V3, V7, Class) |> 
  mutate(Class = factor(Class, levels = c("Rad.Flow", "High")))

# Define a custom theme CSS
custom_css <- "
body {
  background-color: #ecf0f1;
  font-family: 'Helvetica Neue', sans-serif;
  font-size: 14px;
}

.header, .button {
  background-color: #3498db;
  color: #fff;
  font-size: 24px;
  font-weight: bold;
  padding: 10px;
}

.button {
  border: none;
  cursor: pointer;
}

.button:hover {
  background-color: #2980b9;
}

.sidebar, .sidebar a {
  background-color: #2980b9;
  color: #fff;
}

.sidebar a {
  text-decoration: none;
  display: block;
  padding: 10px 20px;
}

.sidebar a:hover {
  background-color: #3498db;
}

.panel, .plot, .text-input, .select-input, .text-output {
  background-color: #fff;
  border: 1px solid #ecf0f1;
  padding: 20px;
  margin: 10px;
}

.text-input, .select-input {
  padding: 10px;
  width: 100%;
}

.slider {
  width: 100%;
}

.accordion, .panel-heading, .panel-title, .panel-body {
  background-color: #fff;
  border: 1px solid #ecf0f1;
  margin: 10px;
}

.panel-heading {
  background-color: #3498db;
  color: #fff;
  padding: 10px 20px;
}

.panel-title {
  font-size: 16px;
  font-weight: bold;
}

.panel-body {
  padding: 20px;
}
"

# User Interface
ui = page_fillable(
  shinyjs::useShinyjs(),
  titlePanel(
    div(style = "text-align: center;", tags$b("Comparing Hyperparameter Tuning Methods"))
  ),
  layout_sidebar(
    fillable = FALSE,
    sidebar = sidebar(
      fluidRow(
        column(12, style = "text-align: center;", tags$h3("User Inputs"))),
      fluidRow(
        checkboxInput(
          "use_default_data",
          "Use Example Dataset (Subset of \"Shuttle\" from mlbench)",
          TRUE
        )
      ),
      conditionalPanel(
        condition = "!input.use_default_data",
        fluidRow( # Source: https://shiny.posit.co/r/articles/build/upload/
          column(8, fileInput(
            "data_file",
            "Upload CSV File",
            multiple = FALSE,
            accept = c("text/csv","text/comma-separated-values,text/plain",".csv")
          ))
        ),
        fluidRow(
          column(4, checkboxInput(
            "header",
            "Includes Headers",
            TRUE
          )),
          column(4, selectInput(
            "sep",
            "Separator Type",
            choices = c("Comma" = ",", "Semicolon" = ";", "Tab" = "\t"),
            selected = ","
          )),
          column(4, selectInput(
            "quote",
            "Quote Type",
            choices = c("None" = "", "Double Quote" = '"', "Single Quote" = "'"),
            selected = '"'
          ))
        ),
        fluidRow(
          column(4, actionButton("upload_data", "Upload"))
        )
      ),
      tags$hr(),
      fluidRow(
        column(6, uiOutput("n_slider")),
        column(6, selectInput(
          "algorithm",
          "Select Model Type",
          c("Random Forest" = "random_forest", "LASSO" = "LASSO", "Ridge" = "ridge")
        ))
      ),
      fluidRow(
        column(4, uiOutput("task_selector")),
        column(4, uiOutput("outcome_selector")),
        column(4, uiOutput("metric_selector"))
      ),
      accordion(
        accordion_panel(
          HTML("<b>Grid/Random Search Space</b>"),
          sliderInput("random_n", "Proportion of Space to Random Search", 
                      min = 0.1, max = 1, value = .5),
          conditionalPanel(
            condition = "input.algorithm == 'LASSO' || input.algorithm == 'ridge'",
            numericInput("lambda_min", "Penalty Value (Min)", value = 0),
            numericInput("lambda_max", "Penalty Value (Max)", value = 1),
            numericInput("lambda_levels", "Penalty Value (# Levels)", value = 10)
          ),
          conditionalPanel(
            condition = "input.algorithm == 'random_forest'",
            fluidRow(
              column(4, numericInput("mtry_min", "Predictors per Split (Min)", value = 1)),
              column(4, numericInput("mtry_max", "Predictors per Split (Max)", value = 3)),
              column(4, numericInput("mtry_levels", "Predictors per Split (# Levels)", value = 3))
            ),
            fluidRow(
              column(4, numericInput("min_n_min", "Min Obs. Terminal Nodes (Min)", value = 10)),
              column(4, numericInput("min_n_max", "Min Obs. Terminal Nodes (Max)", value = 100)),
              column(4, numericInput("min_n_levels", "Min Obs. Terminal Nodes (# Levels)", value = 3))
            ),
            fluidRow(
              column(4, numericInput("trees_min", "Trees (Min)", value = 10)),
              column(4, numericInput("trees_max", "Trees (Max)", value = 100)),
              column(4, numericInput("trees_levels", "Trees (# Levels)", value = 3))
            )
          )
        ),
        accordion_panel(
          HTML("<b>Bayesian Optimization</b>"),
          fluidRow(
            column(6, numericInput("bayes_init", "Sets of Initial Hyperparameters", value = 5)),
            column(6, numericInput("bayes_iter", "Bayes Iterations", value = 5))
          )
        ),
        actionButton("start_search", "Start search")
      ),
      width = 600
    ),
    fluidRow(
      column(12, style = "text-align: center;", tags$h3("Optimal Hyperparameters")),
      column(4, p(style = "font-size: 16px; font-weight: bold;", "Under Grid Search")),
      column(4, p(style = "font-size: 16px; font-weight: bold;", "Under Random Search")),
      column(4, p(style = "font-size: 16px; font-weight: bold;", "Under Bayesian Optimization"))
    ),
    fluidRow(
      column(4, verbatimTextOutput("grid_hyperparams")),
      column(4, verbatimTextOutput("random_hyperparams")),
      column(4, verbatimTextOutput("bayes_hyperparams"))
    ),
    mainPanel(
      conditionalPanel(
        condition = "input.start_search > 0",
        style = "display: none;",
        withSpinner(plotlyOutput("metric_plot"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Define list of reactive values to persist
  v = reactiveValues()
  
  # Set default data whenever user selects default data checkbox
  v$data = shuttle_filtered
  observeEvent(input$use_default_data, {v$data = shuttle_filtered})
  
  # Allow user to upload data
  observeEvent(c(input$data_file, input$upload_data), {
    req(input$data_file, input$upload_data)
    v$data = read.csv(
      input$data_file$datapath,
      header = input$header,
      sep = input$sep,
      quote = input$quote
    )
  })
  
  # Create slider to subset number of observations from data
  output$n_slider = renderUI({
    sliderInput(
      "n_slider",
      "Select # of Observations",
      min = dim(v$data)[1] %/% 10,
      max = dim(v$data)[1],
      value = dim(v$data)[1],
      ticks = FALSE
    )
  })
  
  # Algorithm -> Task -> Outcome Var -> Metric
  # Specify potential task types depending on algorithm selection
  tasks = reactive({
    req(input$algorithm)
    
    if (input$algorithm == "random_forest") {
      return(c("Classification" = "classification",
               "Regression" = "regression"))
    } else {
      return(c("Regression" = "regression"))
    }
  })
  
  output$task_selector = renderUI({
    selectInput("task", "Select Task", tasks())
  })
  
  # Specify potential outcome variables depending on task selection
  outcome_vars = reactive({
    req(input$task)
    
    if (input$task == "regression"){
      numeric_vars = names(Filter(is.numeric, v$data))
      if (length(numeric_vars) == 0){
        return("Error: 0 numeric vars")
      } else {
        return(str_sort(numeric_vars, numeric = TRUE))
      }
    } else if (input$task == "classification"){
      factor_vars = names(Filter(is.factor, v$data))
      if (length(factor_vars) == 0){
        return("Error: 0 factor vars")
      } else {
        return(str_sort(factor_vars, numeric = TRUE))
      }
    }
  })
  
  output$outcome_selector = renderUI({
    options = outcome_vars()
    selectInput("outcome_var", "Select Outcome", options, selected = options[1])
  })
  
  # Specify potential metrics depending on outcome_var selection
  metrics = reactive({
    req(input$task, input$outcome_var)
    if (input$outcome_var %in% c("Error: 0 factor vars", "Error: 0 numeric vars")){
      return("")
    }
    if (input$task == "classification") {
      if (length(unique(v$data[, input$outcome_var])) > 2){
        return(c(
          "Accuracy" = "accuracy"
        ))
      }
      if (length(unique(v$data[, input$outcome_var])) == 2){
        return(c(
          "Accuracy" = "accuracy", 
          "Precision" = "precision",
          "Recall" = "recall",
          "ROC_AUC" = "roc_auc",
          "F1 Score" = "f_meas"
        ))
      }
      if (length(unique(v$data[, input$outcome_var])) < 2){
        return("Selected outcome variable has < 2 unique classes.")
      }
    } else if (input$task == "regression") {
      return(c(
        "RMSE" = "rmse",
        "MAE" = "mae"
      ))
    } else {
      return("Selected task is not currently supported.")
    }
  })
  
  output$metric_selector = renderUI({
    selectInput("metric", "Select Metric", metrics())
  })
  
  # Create discrete hyperparameter space
  hyperparam_space = eventReactive(input$start_search, {
    if(input$algorithm == "random_forest"){
      hyperparam_space = grid_regular(
        mtry(c(input$mtry_min, input$mtry_max)),
        min_n(c(input$min_n_min, input$min_n_max)),
        trees(c(input$trees_min, input$trees_max)),
        levels = c(mtry = input$mtry_levels, min_n = input$min_n_levels, trees = input$trees_levels)
      )
    } else if (input$algorithm %in% c("LASSO", "ridge")){
      hyperparam_space = grid_regular(
        penalty(as.double(c(input$lambda_min, input$lambda_max)), trans = NULL),
        levels = input$lambda_levels
      )
    }
    return(hyperparam_space)
  })
  
  observeEvent(input$bayes_init, {
    req(input$bayes_init)
    if (input$bayes_init < 4) {
      updateNumericInput(session, "bayes_init", value = 4)
    }
  })
  
  
  v$grid_best = reactiveVal(NULL)
  v$random_best = reactiveVal(NULL)
  v$bayes_best = reactiveVal(NULL)
  
  # Run search
  observeEvent(input$start_search, {
    req(input$start_search)
    shinyjs::disable("start_search")
    
    # get hyperparameter space
    hyperparam_space = hyperparam_space()
    print(hyperparam_space)
    
    # modify dataset to size n
    sampled_data = sample_n(v$data, input$n_slider, replace = FALSE)
    
    # If LASSO or ridge, remove non-numeric variables
    if (input$algorithm %in% c("LASSO", "ridge")){
      sampled_data = Filter(is.numeric, sampled_data)
    }
    
    # create empty data frame to store search results
    convergence = data.frame(
      seconds_passed = numeric(0),
      best_score = numeric(0),
      tuning_type = character(0)
    )
    
    for (tune_type in c("grid", "random", "bayes")) {
      # start timer object in shiny #TODO
      # start internal timer
      start.time = Sys.time()
      
      # Define tunetype-specific parameters for tune_model function
      if (tune_type == "grid") {
        tuning_grid = hyperparam_space
        initial_input = NULL
        bayes_iter = NULL
      } else if (tune_type == "random"){
        tuning_grid = sample_n(
          hyperparam_space, 
          round(dim(hyperparam_space)[[1]]*input$random_n),
          replace = FALSE
        )
        initial_input = NULL
        bayes_iter = NULL
      } else {
        tuning_grid = NULL
        initial_input = input$bayes_init
        bayes_iter = input$bayes_iter
      }
      
      # Train and tune model
      tuning_results = tune_model(
        sampled_data,
        input$outcome_var,
        input$algorithm,
        input$task,
        input$metric,
        tune_type,
        tuning_grid = tuning_grid,
        initial_input = initial_input,
        bayes_iter = bayes_iter
      )
      
      # end timer
      end.time = Sys.time()
      elapsed.time = as.numeric(difftime(end.time, start.time, units = "secs"))
      
      estimates = tuning_results[[1]]
      best = tuning_results[[2]]
      
      if (tune_type != "bayes") {
        estimates = estimates %>%
          mutate(.iter = 1:dim(estimates)[[1]])
        if (tune_type == "grid") {
          v$grid_best(best)
        } else if (tune_type == "random") {
          v$random_best(best)
        }
      } else {
        v$bayes_best(best)
      }
      
      # convert estimates to tall dataframe to use ultimately for plotting
      estimates = estimates %>% rename(iteration = .iter) %>% filter(iteration > 0)
      
      # calculate seconds per iteration to use for plotting
      iters = dim(estimates)[[1]]
      time_per_iter = round(elapsed.time / iters, 2)
      
      if (input$task == "classification") {
        iter_scores = estimates %>%
          mutate(best_score = sapply(1:iters, function(iter) {
            max(estimates$mean[1:iter])}),
            seconds_passed = iteration * time_per_iter,
            tuning_type = tune_type
          ) %>%
          select(seconds_passed, best_score, tuning_type)
      } else {
        iter_scores = estimates %>%
          mutate(best_score = sapply(1:iters, function(iter) {
            min(estimates$mean[1:iter])}),
            seconds_passed = iteration * time_per_iter,
            tuning_type = tune_type
          ) %>%
          select(seconds_passed, best_score, tuning_type)
      }
      
      # row bind to larger dataframe for all iterations
      convergence = rbind(convergence, iter_scores)
      
      # output best parameters in shiny app
      if (input$algorithm == "random_forest") {
        # update shiny app so that best parameters show in value boxes
        if (tune_type == "grid") {
          output$grid_hyperparams = renderText({
            best_grid = v$grid_best()
            if (!is.null(best_grid)) {
              paste0(
                "Variables per Split: ", round(best_grid[1,"mtry"], 3), "\n",
                "Min Obs Terminal Nodes: ", round(best_grid[1,"min_n"], 3), "\n",
                "Number of Trees: ", round(best_grid[1,"trees"], 3), "\n"
              )
            }
          })
        } else if (tune_type == "random") {
          output$random_hyperparams = renderText({
            best_random = v$random_best()
            if (!is.null(best_random)) {
              paste0(
                "Variables per Split: ", round(best_random[1,"mtry"], 3), "\n",
                "Min Obs Terminal Nodes: ", round(best_random[1,"min_n"], 3), "\n",
                "Number of Trees: ", round(best_random[1,"trees"], 3), "\n"
              )
            }
          })
        } else {
          output$bayes_hyperparams = renderText({
            best_bayes = v$bayes_best()
            if (!is.null(best_bayes)) {
              paste0(
                "Variables per Split: ", round(best_bayes[1,"mtry"], 3), "\n",
                "Min Obs Terminal Nodes: ", round(best_bayes[1,"min_n"], 3), "\n",
                "Number of Trees: ", round(best_bayes[1,"trees"], 3), "\n"
              )
            }
          })
        }
      } else {
        if (tune_type == "grid") {
          output$grid_hyperparams = renderText({
            best_grid = v$grid_best()
            if (!is.null(best_grid)) {
              paste0("Penalty: ", round(best[1,"penalty"], 3))
            }
          })
        } else if (tune_type == "random") {
          output$random_hyperparams = renderText({
            best_random = v$random_best()
            if (!is.null(best_random)) {
              paste0("Penalty: ", round(best_random[1,"penalty"], 3))
            }
          })
        } else {
          output$bayes_hyperparams = renderText({
            best_bayes = v$bayes_best()
            if (!is.null(best_bayes)) {
              paste0("Penalty: ", round(best_bayes[1,"penalty"], 3))
            }
          })
        }
      }
    }
    
    # create plot
    output$metric_plot = renderPlotly({
      p = ggplot(
        convergence,
        aes(x = seconds_passed, y = best_score, color = tuning_type)
      ) +
        geom_line() +
        labs(
          x = "Elapsed Time (seconds)",
          y = glue("Metric Value"),
          color = "Tuning Type"
        ) +
        theme(
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10)
        )
      ggplotly(p)
    })
    
    # Renable start button to re-run search
    shinyjs::enable("start_search")
    
  })
  
}

shinyApp(ui, server)