# HyperparameterTuning

This Shiny app allows users to quickly compare the efficacy of three common hyperparameter tuning methods: Grid Search, Random Search, and Bayesian Optimization.

## Quick Start

This Shiny app allows users to quickly compare the efficacy of three common hyperparameter tuning methods: Grid Search, Random Search, and Bayesian Optimization. This section describes the basic information you need to get started.

#### Launching the App

To launch the R Shiny app, start a local or cloud-based instance of RStudio and open the `ShinyApp.R` file. Select the `Run App` option in RStudio.

#### Configuring Inputs

1. Dataset Selection:

    *Example Dataset*: You have the option to use a provided example dataset within the application. The dataset is a subset of the 'Shuttle' dataset from the mlbench package, and is pre-loaded and ready for analysis.

    *Custom Data Upload*: Alternatively, you can upload your own CSV file for a tailored analysis. Note that the data you upload should already be preprocessedvand ready for modeling (e.g., categorical variables already one-hot encoded).
    If you are interested in performing classification, the outcome variable in your data should be a factor variable; if regression, the outcome variable should be numeric. 
    
    Additionally, if you opt to test your own dataset, ensure you configure the dataset settings, including specifying headers and separators, to align with your data. Note that you must click the 'Upload' button after selecting the file name and other settings.
  
2. Model Configuration:

    *Machine Learning Algorithm*: Select the machine learning algorithm that best suits your analysis. Options include LASSO, Ridge, or Random Forest.

    *Task Type*: Define the nature of your task by choosing either Regression or Classification, depending on the problem you are addressing.

    *Outcome Variable*: Specify the target outcome variable from your dataset that you intend to predict.

    *Evaluation Metric*: Choose the evaluation metric that aligns with your performance objectives, ensuring that the model is optimized for your specific goals.

3. Hyperparameter Space:

    *Grid/Random Search Space*: Customize the hyperparameter search space for grid and random search to align with the chosen algorithm's requirements.

    - Random Forest Parameters: For Random Forest, configure the ranges for key hyperparameters, such as the number of predictors per split, minimum observations in terminal nodes, and the number of trees.
    
    - LASSO/Ridge: Configure the range and number of intermediate values to test for the penalty hyperparameter.

    *Bayesian Optimization Settings*: For Bayesian Optimization, specify the number of initial hyperparameter sets and define the number of Bayesian iterations for the optimization process.

#### Understanding the Outputs

1. Feedback:

    The current version of the Shiny app runs all three tuning methods (Grid Search, Random Search, and Bayesian Optimization) before displaying the results. Future versions of the app will provide real-time feedback on the best metric score and associated hyperparameters discovered by each of the three tuning methods.

2. Convergence Visualization:

    The application features an interactive plot that allows you to analyze the convergence of each tuning method over time. The plot displays metric scores on the y-axis and time elapsed on the x-axis, providing a clear visual representation of the efficiency and effectiveness of each approach. Metric scores are computed as the average score from 5-fold cross validation.
    
    Note on interpretation: 
    
    - Grid Search will typically match or outperform Random Search in terms of metric score, since Random Search only uses a subset of the hyperparameter space used for Grid Search; however, Random Search may achieve a near-optimal metric score in a fraction of the time required for Grid Search.
    
    - Bayesian Optimization will almost always take longer than Grid and Random Search when the hyperparameter space is relatively small (as in the example shown in the image above); however, when the space is large, Bayesian Optimization may converge to the optimal hyperparameters faster than Grid and Random Search.

3. Optimal Hyperparameters:

For your convenience and comparison, the optimal hyperparameters found by each tuning method (Grid Search, Random Search, Bayesian Optimization) are displayed in separate sections within the application. This allows you to quickly assess and utilize the best hyperparameters for your machine learning model.
  
This Shiny application serves as a tool for data scientists and machine learning practitioners, enabling them to experiment with different hyperparameter tuning methods and make informed decisions about which approach is most suitable for their specific datasets and tasks. It promotes efficient model optimization and enhances predictive performance by automating the hyperparameter tuning process and visualizing its progress.

## Detailed Methodology

### 1. Overview and Purpose

Selecting optimal hyperparameters for a machine learning model is a fundamental step in achieving the highest predictive performance. This process involves exploring a vast hyperparameter space to discover the most effective values. Broadly speaking, there are two categories of strategies for this exploration: uninformed and informed search methods.

Uninformed search methods systematically traverse the hyperparameter space, lacking the ability to incorporate insights gained during the search. One widely used uninformed search method is Grid Search, which involves exhaustively evaluating predefined hyperparameter combinations, covering a range of values for each hyperparameter. This approach provides a comprehensive overview of the space but can be computationally expensive. Another uninformed method is Random Search, which, as the name suggests, randomly samples hyperparameter combinations from the space. While it may seem less systematic than Grid Search, Random Search has shown its effectiveness in finding good hyperparameters with fewer evaluations.

In contrast, informed search methods utilize heuristics to navigate the space more efficiently, reducing the number of iterations required to find optimal (or near-optimal) hyperparameter values. One notable informed search method is Bayesian Optimization, which leverages probabilistic models to guide the search towards promising regions of the hyperparameter space. By learning from previous evaluations, Bayesian Optimization adapts its search strategy and efficiently narrows down the search space. However, this efficiency comes at a cost, as informed search methods demand more time and computational resources due to the evaluation of heuristics after each iteration.

The choice between uninformed and informed search methods depends on factors such as the user's data, task, and the specific machine learning model employed. In light of these considerations, this project aims to develop a practical tool that empowers users to experiment with these three prevalent hyperparameter tuning methods: Grid Search, Random Search, and Bayesian Optimization. The ultimate objective is to help users determine which method is likely to yield the best results while adhering to constraints on time and computational resources.

### 2. Shiny App for Hyperparameter Tuning Comparison

In this section, we introduce a Shiny application designed for comparing and evaluating different hyperparameter tuning methods: Grid Search, Random Search, and Bayesian Optimization. This interactive tool allows users to explore and analyze the performance of these methods in the context of machine learning model hyperparameter tuning. The primary objective of this Shiny app is to provide users with a convenient means to assess the effectiveness of various hyperparameter tuning strategies. Users can utilize the app to:

1. Select Data: Choose between using an example dataset or uploading their own CSV file, providing flexibility in dataset selection.

2. Specify Model Parameters: Configure key settings, including the choice of machine learning algorithm (LASSO, Ridge, or Random Forest), the type of task (Regression or Classification), the target outcome variable, and the evaluation metric.

3. Define Hyperparameter Space: Set the hyperparameter search space tailored to the selected algorithm. Users can specify hyperparameters such as the penalty values for LASSO and Ridge, and for Random Forest, parameters like the number of predictors per split, minimum observations in terminal nodes, and the number of trees.

4. Hyperparameter Tuning Methods: Evaluate the performance of hyperparameter tuning methods:
  Grid Search: A systematic exploration of predefined hyperparameter combinations.
  Random Search: A randomized sampling of hyperparameter combinations.
  Bayesian Optimization: A probabilistic model-based method that adapts its search strategy based on previous evaluations.

5. Start Search: Initiate the hyperparameter tuning process to visualize the convergence of each method over time and identify the optimal hyperparameters selected by each tuning method.


### 3. Code Explanation

In this section, we discuss the rationale behind the code implementation for the Shiny app.

1. tune_model Function:

We first started with defining the function `tune_model` in a separate R file. The `tune_model` function is a versatile function that automates the process of hyperparameter tuning and model evaluation for machine learning tasks. It takes several crucial inputs, including the dataset, target outcome variable, machine learning algorithm, task type, metric function, and the type of hyperparameter tuning method (Grid Search, Random Search, or Bayesian Optimization). This function first prepares the dataset by creating a model formula and splitting it into training and testing sets. It also sets up k-fold cross-validation for robust model evaluation. Depending on the selected algorithm and task type, the function configures an appropriate machine learning model and defines a workflow that includes preprocessing steps. It then conducts hyperparameter tuning using either grid search, random search or Bayesian optimization, fine-tuning model hyperparameters to optimize the chosen evaluation metric. Finally, it returns valuable outputs, including performance estimates over the tuning process and the best hyperparameters discovered for the specified machine learning algorithm and task. This function significantly modularizes and streamlines the hyperparameter tuning and model selection process, making it an essential tool for efficient model development and optimization.

2. User Interface: 

The user interface (UI) includes a sidebar layout that allows users to set all require inputs in one section. First, the user can choose whether to use the example dataset provided or upload their own CSV/text file, in which case they are prompted to select the file location (and other settings) before uploading. They can then select whether to use a subset of the total observations, as well as the model type, task, outcome variable, and metric. These inputs are dynamic; that is, the choice of model type affects the task options, and the choice of task affects the options for outcome variable and metric.

Collapsible panel layout sections allow users to set the inputs required to define the Grid/Random Search Space and for Bayesian Optimization (as these methods require different types of inputs). The "Grid/Random Search Space" section includes a `sliderInput` for adjusting the proportion of the search space used for random search. It features two `conditionalPanel` elements, which display different UI inputs based on the selected algorithm (either LASSO/Ridge or Random Forest). These inputs specify the range and levels of various hyperparameters, such as the penalty value for LASSO/Ridge, and the number of predictors per split, minimum observations in terminal nodes, and the number of trees for Random Forest. The "Bayesian Optimization" section contains `numericInput` fields for setting the number of initial sets of hyperparameters and the number of iterations for Bayesian optimization. An `actionButton` at the end enables users to initiate the search or optimization process.

The main panel compares and visualizes the performance of three hyperparameter tuning methods: Grid Search, Random Search, and Bayesian Optimization. The plot displaying the metric scores by
elapsed time is implemented with Plotly to allow for the user to zoom, pan, and otherwise interact with the plot.

3. Server Logic:

The server function begins by initializing a list of reactive values, `v`, which is crucial for storing and managing the state within the app. The default `Shuttle` dataset is added to that
list and overwritten if the user uploads their own data. When the user clicks the "Upload" button, the app reads the uploaded file using `read.csv` and updates the `v$data` reactive value with the new data.

The application dynamically generates UI elements based on user interactions. For example, a slider input is created for users to select the number of observations from the current dataset, with its range set according to the size of `v$data`. The UI elements also adapt depending on the algorithm selected by the user. Different options for tasks, outcome variables, and metrics are provided based on both the algorithm and the task type selected. This adaptability extends to the specification of hyperparameters, where `conditionalPanel` is used to display different inputs for LASSO/Ridge and Random Forest algorithms.

The core functionality of the server is the hyperparameter tuning process. Upon the user clicking the "Start search" button, the app constructs a discrete hyperparameter space for grid and random search based on the user's inputs. The app then proceeds with hyperparameter tuning using the `tune_model` function
located in the `models.R` file. The arguments passed to the function are tailored according to the selected algorithm, task, metric, and tuning type. The models are trained and the best metric results and hyperparameters are recorded after each iteration. The results of the tuning process are visualized using ggplot and Plotly, and the optimal hyperparameters for each method are displayed in the main panel of the app as text outputs.

### 4. Discussion & Conclusion

In the current iteration of our Shiny app, we have created a user-friendly platform that facilitates the exploration of hyperparameter tuning methods in machine learning. The app's dynamic UI elements, capable of adapting based on user input, make it accessible and practical for a range of users. It supports key algorithms like LASSO, Ridge, and Random Forest, and offers flexibility in data handling, allowing users to upload their datasets or use predefined ones. This versatility is especially beneficial for comparative analysis between different tuning methods such as Grid, Random, and Bayesian Optimization. However, the app's current limitations include its restricted range of machine learning models, lack of real-time updating functionality, slow model training speed, and lack of robust user input validation checks.

Expanding the app to include a broader array of models, such as Support Vector Machines (SVM), Gradient Boosting (GBM), K-Nearest Neighbors (KNN), or Neural Networks is a crucial next step. This addition would not only enhance the app's comprehensiveness but also cater to a more diverse user base, particularly those dealing with complex data sets where advanced modeling techniques are imperative. Incorporating other algorithms would provide users with deeper insights into the suitability and performance of various models for different data types. Additionally, improving data visualization and introducing more interactive elements could significantly enrich the user experience.
Ideally, the app would provide the user with real-time feedback about model training, including information about the current iteration, the specific hyperparameters under test, and the best metric score so far for each of the three tuning methods. We would also prefer that the three tuning methods run concurrently so that users can compare results across methods in real-time. However, implementing the tuning methods from scratch was feasible for grid and random search but proved much more challenging for Bayesian Optimization. The reason is that Bayesian Optimization relies on probabilistic models to predict the performance of untested hyperparameters, which requires complex mathematical computations and may not easily align with the real-time feedback requirements of the application. The tidymodels framework does not provide an obvious way to extract the results from each iteration in real-time. Additionally, while many model-tuning functions utilize multiple CPU cores for parallel processing, allocating these resources for concurrent updates across tuning methods introduces complexities in resource management and coordination, making it a non-trivial task.

In general, the app would also benefit substantially from efforts to optimize model training to increase the speed of hyperparameter tuning. Currently, relatively small hyperparameter spaces (e.g., 50 to 100 combinations) can be searched using all three tuning methods within 10 to 15 minutes. However, users will likely want to test much larger spaces, and therefore optimization of the model training and tuning functions would likely be required.

Finally, we have implemented basic checks to validate user inputs, such as confirming that any user-provided data has at least one factor (numeric) variable to use as the outcome variable for classification (regression) tasks. However, given the complexity of
all possible datasets, algorithms, tasks, and metrics that a user can select, more robust user input validation is needed.

Given the constraints of time and resources, addressing these limitations was not feasible for the initial version of the app. However, with further development and user feedback, we aim to evolve the app into a more versatile and powerful tool. This expansion, focused on user needs and the latest trends in machine learning, will ensure the app remains a valuable asset for both educational and practical applications in the field.
