"""Definitions of specific API points"""

from flask_restx import fields

definitions = {"regression": {}, "classification": {}, "clustering": {}, "outlier": {}}
"""Definitions of the restplus fields for each model"""

ui_to_inner = {"forest": "random-forest", "gradient": "gradient-boosting"}
"""Mapping irregular API REST names to internal package names"""

ui_to_friendly = {"tree": "decision tree", "forest": "random forest", "gradient": "gradient boosting",
                  "linear": "linear", "logistic": "logistic regression", "mlp": "multilayer perceptron",
                  "bayes": "gaussian naive Bayes", "sv": "support vector", "neighbors": "k-nearest neighbors",
                  "extra-trees": "Extremely randomized trees",
                  # Clustering
                  "kmeans": "k-means", "dbscan": "DBSCAN", "gaussianmixture": "Gaussian Mixture",
                  "spectral": "Spectral Clustering", "agglomerative": "Agglomerative Clustering",
                  }
"""Mapping from API REST names to user-friendly names"""

scaling_methods = {
    'method': fields.String(
        description='Method used to scale numeric features before a model is trained. Available options and their '
                    'specific configuration parameters are the following:\n'
                    '- standard: Scale to center the mean and unitarize variance. Can be configured to do only one of '
                    'the steps, e.g.,  {"with_mean": true, "with_std": false}.\n'
                    '- maxabs: Scale so the maximal absolute value is set to 1.0.\n'
                    '- minmax: Scale to fit in the range (0, 1) or any other given in the feature_range parameter '
                    '(e.g., {"feature_range": (-1,1)}).\n'
                    '- robust: Scale using quantiles, which are robust statistical indicators. Sets the median to 0 '
                    'and a given quantile range to 1 (defaults to the interquartile range). Can be further configured '
                    'using parameters, e.g., {"with_centering": true, "with_scaling": true, '
                    'quantile_range": (25.0, 75.0)}.',
        enum=["standard", "maxabs", "minmax", "robust"],
        example="standard"
    ),
    'pars': fields.Raw(
        description='Additional parameters to further configure the scaling. See the method description.')

}

# Decision tree
definitions["regression"]["tree"] = {
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, and “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node.",
        enum=["mse", "friedman_mse", "mae"],
        example="mse"
    ),
    'splitter': fields.String(
        description="The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.",
        enum=["best", "random"],
        example="best"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=10
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
    'presort': fields.Boolean(
        description="Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. When using either a smaller dataset or a restricted depth, this may speed up the training.",
        example=False
    ),
}

definitions["classification"]["tree"] = {
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.",
        enum=["gini", "entropy"],
        example="mse"
    ),
    'splitter': fields.String(
        description="The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.",
        enum=["best", "random"],
        example="best"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=10
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
    'presort': fields.Boolean(
        description="Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. When using either a smaller dataset or a restricted depth, this may speed up the training.",
        example=False
    ),
}

# Random forest
definitions["regression"]["forest"] = {
    'n_estimators': fields.Integer(
        description="The number of trees in the forest.",
        example=10
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.",
        enum=["mse", "mae"],
        example="mse"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=100
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'bootstrap': fields.Boolean(
        description="Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.",
        example=True
    ),
    'oob_score': fields.Boolean(
        description="Whether to use out-of-bag samples to estimate the R^2 on unseen data.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

definitions["classification"]["forest"] = {
    'n_estimators': fields.Integer(
        description="The number of trees in the forest.",
        example=100
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.",
        enum=["gini", "entropy"],
        example="gini"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=10
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'bootstrap': fields.Boolean(
        description="Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.",
        example=True
    ),
    'oob_score': fields.Boolean(
        description="Whether to use out-of-bag samples to estimate the R^2 on unseen data.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

definitions["classification"]["extra-trees"] = {
    'n_estimators': fields.Integer(
        description="The number of trees in the forest.",
        example=100
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.",
        enum=["gini", "entropy"],
        example="gini"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=10
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'bootstrap': fields.Boolean(
        description="Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.",
        example=True
    ),
    'oob_score': fields.Boolean(
        description="Whether to use out-of-bag samples to estimate the R^2 on unseen data.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

definitions["regression"]["extra-trees"] = {
    'n_estimators': fields.Integer(
        description="The number of trees in the forest.",
        example=100
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error.",
        enum=["mse", "mae"],
        example="mse"
    ),
    'max_depth': fields.Integer(  # TODO: Allow None
        description="The maximum depth of the tree.",
        example=10
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'bootstrap': fields.Boolean(
        description="Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.",
        example=True
    ),
    'oob_score': fields.Boolean(
        description="Whether to use out-of-bag samples to estimate the R^2 on unseen data.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

# Gradient boosting
definitions["regression"]["gradient"] = {
    'loss': fields.String(
        description="Loss function to be optimized. ‘ls’ refers to least squares regression. ‘lad’ (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. ‘huber’ is a combination of the two. ‘quantile’ allows quantile regression (use alpha to specify the quantile).",
        enum=["ls", "lad", "huber", "quantile"],
        example="ls"
    ),
    'alpha': fields.Float(
        description="The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.",
        example=0.9
    ),
    'learning_rate': fields.Float(
        description="Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.",
        example=0.1
    ),
    'n_estimators': fields.Integer(
        description="The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.",
        example=100
    ),
    'subsample': fields.Float(
        description="The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.",
        example=1.0
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.",
        enum=["friedman_mse", "mse", "mae"],
        example="friedman_mse"
    ),
    'max_depth': fields.Integer(
        description="Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.",
        example=3
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'max_leaf_nodes': fields.Integer(
        description="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        example=100
    ),
    'n_iter_no_change': fields.Integer(
        description="used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations.",
        example=100
    ),
    'validation_fraction': fields.Float(
        description="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.",
        example=0.1
    ),
    'tol': fields.Float(
        description="Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.",
        example=0.1
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

definitions["classification"]["gradient"] = {
    'loss': fields.String(
        description="Loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.",
        enum=["deviance", "exponential"],
        example="deviance"
    ),
    'learning_rate': fields.Float(
        description="Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.",
        example=0.1
    ),
    'n_estimators': fields.Integer(
        description="The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.",
        example=100
    ),
    'subsample': fields.Float(
        description="The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.",
        example=1.0
    ),
    'criterion': fields.String(
        description="The function to measure the quality of a split. Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.",
        enum=["friedman_mse", "mse", "mae"],
        example="friedman_mse"
    ),
    'max_depth': fields.Integer(
        description="Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.",
        example=3
    ),
    'min_samples_split': fields.Integer(
        description="The minimum number of samples required to split an internal node.",
        example=2
    ),
    'min_samples_leaf': fields.Integer(
        description="The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        example=2
    ),
    'max_features': fields.Integer(
        description="The number of features to consider when looking for the best split.",
        example=20
    ),
    'max_leaf_nodes': fields.Integer(
        description="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        example=100
    ),
    'n_iter_no_change': fields.Integer(
        description="used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations.",
        example=100
    ),
    'validation_fraction': fields.Float(
        description="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.",
        example=0.1
    ),
    'tol': fields.Float(
        description="Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.",
        example=0.1
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
}

# Multilayer perceptron
definitions["regression"]["mlp"] = {
    'hidden_layer_sizes': fields.List(fields.Integer,
                                      description="The ith element represents the number of neurons in the ith hidden layer.",
                                      example=[100, ]),
    'activation': fields.String(
        description="""Activation function for the hidden layer.
    ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
    ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
    ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
    ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)""",
        enum=["identity", "logistic", "tanh", "relu"],
        example="relu"
    ),
    'solver': fields.String(
        description="""The solver for weight optimization.

    ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
    ‘sgd’ refers to stochastic gradient descent.
    ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
    Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.""",
        enum=["lbfgs", "sgd", "adam"],
        example="adam"
    ),
    'alpha': fields.Float(
        description="L2 penalty (regularization term) parameter.",
        example=0.0001
    ),
    'learning_rate': fields.String(
        description="""Learning rate schedule for weight updates.

    ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
    ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
    ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.""",
        enum=["constant", "invscaling", "adaptive"],
        example="constant"
    ),
    'learning_rate_init': fields.Float(
        description="The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.",
        example=0.001
    ),
    'power_t': fields.Float(
        description="The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.",
        example=0.5
    ),
    'max_iter': fields.Integer(
        description="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.",
        example=200
    ),
    'shuffle': fields.Boolean(
        description="Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
    'tol': fields.Float(
        description="Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.",
        example=0.0001
    ),
    'momentum': fields.Float(
        description="Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.",
        example=0.9
    ),
    'nesterovs_momentum': fields.Boolean(
        description="Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.",
        example=True
    ),
    'early_stopping': fields.Boolean(
        description="Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’",
        example=False
    ),
    'validation_fraction': fields.Float(
        description="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True",
        example=0.1
    ),
    'beta_1': fields.Float(
        description="Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’",
        example=0.9
    ),
    'beta_2': fields.Float(
        description="Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’",
        example=0.999
    ),
    'epsilon': fields.Float(
        description="Value for numerical stability in adam. Only used when solver=’adam’",
        example=1E-8
    ),
    'n_iter_no_change': fields.Integer(
        description="Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’",
        example=10
    ),
}

definitions["classification"]["mlp"] = {
    'hidden_layer_sizes': fields.List(fields.Integer,
                                      description="The ith element represents the number of neurons in the ith hidden layer.",
                                      example=[100, ]),
    'activation': fields.String(
        description="""Activation function for the hidden layer.
‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
‘relu’, the rectified linear unit function, returns f(x) = max(0, x)""",
        enum=["identity", "logistic", "tanh", "relu"],
        example="relu"
    ),
    'solver': fields.String(
        description="""The solver for weight optimization.

‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
‘sgd’ refers to stochastic gradient descent.
‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.""",
        enum=["lbfgs", "sgd", "adam"],
        example="adam"
    ),
    'alpha': fields.Float(
        description="L2 penalty (regularization term) parameter.",
        example=0.0001
    ),
    'learning_rate': fields.String(
        description="""Learning rate schedule for weight updates.

‘constant’ is a constant learning rate given by ‘learning_rate_init’.
‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.""",
        enum=["constant", "invscaling", "adaptive"],
        example="constant"
    ),
    'learning_rate_init': fields.Float(
        description="The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.",
        example=0.001
    ),
    'power_t': fields.Float(
        description="The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.",
        example=0.5
    ),
    'max_iter': fields.Integer(
        description="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.",
        example=200
    ),
    'shuffle': fields.Boolean(
        description="Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.",
        example=True
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
    'tol': fields.Float(
        description="Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.",
        example=0.0001
    ),
    'momentum': fields.Float(
        description="Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.",
        example=0.9
    ),
    'nesterovs_momentum': fields.Boolean(
        description="Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.",
        example=True
    ),
    'early_stopping': fields.Boolean(
        description="Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’",
        example=False
    ),
    'validation_fraction': fields.Float(
        description="The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True",
        example=0.1
    ),
    'beta_1': fields.Float(
        description="Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’",
        example=0.9
    ),
    'beta_2': fields.Float(
        description="Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’",
        example=0.999
    ),
    'epsilon': fields.Float(
        description="Value for numerical stability in adam. Only used when solver=’adam’",
        example=1E-8
    ),
    'n_iter_no_change': fields.Integer(
        description="Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’",
        example=10
    ),
}

# Linear regression
definitions["regression"]["linear"] = {
    'fit_intercept': fields.Boolean(
        description="Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).",
        example=True
    ),
}

# Logistic regression (which is actually classification)
definitions["classification"]["logistic"] = {
    'penalty': fields.String(
        description="Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.",
        enum=["l1", "l2", "elasticnet", "none"],
        example="l2"
    ),
    'dual': fields.Boolean(
        description="Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.",
        example=False
    ),
    'tol': fields.Float(
        description="Tolerance for stopping criteria.",
        example=1E-4
    ),
    'C': fields.Float(
        description="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.",
        example=1.0
    ),
    'fit_intercept': fields.Boolean(
        description="Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.",
        example=True
    ),
    'intercept_scaling': fields.Float(
        description="""Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.

Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.""",
        example=1
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator.",
        example=5
    ),
    'solver': fields.String(
        description="""Algorithm to use in the optimization problem.

For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
‘liblinear’ and ‘saga’ also handle L1 penalty
‘saga’ also supports ‘elasticnet’ penalty
‘liblinear’ does not handle no penalty

Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.""",
        enum=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        example="liblinear"
    ),
    'max_iter': fields.Integer(
        description="Maximum number of iterations taken for the solvers to converge.",
        example=100
    ),
    'l1_ratio': fields.Float(
        description="The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'`. Setting ``l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.",
        example=0.5
    ),
}

# Gaussian Naive Bayes
definitions["classification"]["bayes"] = {
    'var_smoothing': fields.Float(
        description="Portion of the largest variance of all features that is added to variances for calculation stability.",
        example=1E-9
    ),
}

# Support vector
definitions["regression"]["sv"] = {
    'kernel': fields.String(
        description="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, or ‘precomputed’.",
        enum=["linear", "poly", "rbf", "sigmoid", "precomputed"],
        example="rbf",
    ),
    'degree': fields.Integer(
        description="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.",
        example=3
    ),
    'gamma': fields.String(  # TODO: Allow float
        description="""Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
""",
        enum=["auto", "scale"]
    ),
    'coef0': fields.Float(
        description="Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.",
        example=0.0
    ),
    'tol': fields.Float(
        description="Tolerance for stopping criterion.",
        example=1E-3
    ),
    'C': fields.Float(
        description="Penalty parameter C of the error term.",
        example=1.0
    ),
    'epsilon': fields.Float(
        description="Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.",
        example=0.1
    ),
    'shrinking': fields.Boolean(
        description="Whether to use the shrinking heuristic.",
        example=True
    ),
    'max_iter': fields.Integer(
        description="Hard limit on iterations within solver, or -1 for no limit.",
        example=-1
    ),
}

definitions["classification"]["sv"] = {
    'kernel': fields.String(
        description="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, or ‘precomputed’.",
        enum=["linear", "poly", "rbf", "sigmoid", "precomputed"],
        example="rbf",
    ),
    'degree': fields.Integer(
        description="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.",
        example=3
    ),
    'gamma': fields.String(  # TODO: Allow float
        description="""Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
""",
        enum=["auto", "scale"]
    ),
    'coef0': fields.Float(
        description="Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.",
        example=0.0
    ),
    'tol': fields.Float(
        description="Tolerance for stopping criterion.",
        example=1E-3
    ),
    'C': fields.Float(
        description="Penalty parameter C of the error term.",
        example=1.0
    ),
    'epsilon': fields.Float(
        description="Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.",
        example=0.1
    ),
    'shrinking': fields.Boolean(
        description="Whether to use the shrinking heuristic.",
        example=True
    ),
    'max_iter': fields.Integer(
        description="Hard limit on iterations within solver, or -1 for no limit.",
        example=-1
    ),
    'random_state': fields.Integer(
        description="The seed used by the random number generator when shuffling the data for probability estimates.",
        example=5
    ),
}

# k-nearest neighbors
definitions["regression"]["neighbors"] = {
    'n_neighbors': fields.Integer(
        description="Number of neighbors to use by default for kneighbors queries.",
        example=5
    ),
    'weight': fields.String(
        description="""weight function used in prediction. Possible values:
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.""",
        enum=['uniform', 'distance'],
        example="uniform"
    ),
    'algorithm': fields.String(
        description="""Algorithm used to compute the nearest neighbors:

‘ball_tree’ will use BallTree
‘kd_tree’ will use KDTree
‘brute’ will use a brute-force search.
‘auto’ will attempt to decide the most appropriate algorithm based on the values used to fit.""",
        enum=['auto', 'ball_tree', 'kd_tree', 'brute'],
        example="auto"
    ),
    'leaf_size': fields.Integer(
        description="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
        example=30
    ),
    'p': fields.Integer(
        description="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.",
        example=2
    ),
    'metric': fields.String(
        description="""The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.""",
        enum=["euclidean",
              "manhattan",
              "chebyshev",
              "minkowski",
              "wminkowski",
              "seuclidean",
              "mahalanobis",
              "haversine",
              "hamming",
              "canberra",
              "braycurtis",
              "jaccard",
              "matching",
              "dice",
              "kulsinski",
              "rogerstanimoto",
              "russellrao",
              "sokalmichener",
              "sokalsneath"],
        example="minkowski"
    ),

}

definitions["classification"]["neighbors"] = {
    'n_neighbors': fields.Integer(
        description="Number of neighbors to use by default for kneighbors queries.",
        example=5
    ),
    'weight': fields.String(
        description="""weight function used in prediction. Possible values:
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.""",
        enum=['uniform', 'distance'],
        example=5
    ),
    'algorithm': fields.String(
        description="""Algorithm used to compute the nearest neighbors:

‘ball_tree’ will use BallTree
‘kd_tree’ will use KDTree
‘brute’ will use a brute-force search.
‘auto’ will attempt to decide the most appropriate algorithm based on the values used to fit.""",
        enum=['auto', 'ball_tree', 'kd_tree', 'brute'],
        example=5
    ),
    'leaf_size': fields.Integer(
        description="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
        example=30
    ),
    'p': fields.Integer(
        description="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.",
        example=2
    ),
    'metric': fields.String(
        description="""The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.""",
        enum=["euclidean",
              "manhattan",
              "chebyshev",
              "minkowski",
              "wminkowski",
              "seuclidean",
              "mahalanobis",
              "haversine",
              "hamming",
              "canberra",
              "braycurtis",
              "jaccard",
              "matching",
              "dice",
              "kulsinski",
              "rogerstanimoto",
              "russellrao",
              "sokalmichener",
              "sokalsneath"],
        example="minkowski"
    ),

}




# K-means
definitions["clustering"]["kmeans"] = {
    'n_clusters': fields.Integer(
        description="The number of clusters to form as well as the number of centroids to generate.",
        example=8,
    ),
    'init': fields.String(
        description="""Method for initialization. Available values are:
- ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
- ‘random’: choose k observations (rows) at random from data for the initial centroids.""",
        enum=["k-means++", "random"],
        example="k-means++"
    ),
    'n_init': fields.Integer(
        description="Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.",
        example=10,
    ),
    'max_iter': fields.Integer(
        description="Maximum number of iterations of the k-means algorithm.",
        example=300,
    ),
    'tol': fields.Float(
        description="Relative tolerance with regards to inertia to declare convergence",
        example=1E-4,
    ),
    'random_state': fields.Integer(
        description="Seed used for the random number generation for centroid initialization.",
        example=5
    ),
}

# DBSCAN
definitions["clustering"]["dbscan"] = {
    'eps': fields.Float(
        description="The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.",
        example=0.1,
    ),
    'min_samples': fields.Integer(
        description="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.",
        example=3,
    ),
    'metric': fields.String(
        description="""he metric to use when calculating distance between instances in a feature array.""",
        enum=["euclidean",
              "manhattan",
              "chebyshev",
              "minkowski",
              "wminkowski",
              "seuclidean",
              "mahalanobis",
              "haversine",
              "hamming",
              "canberra",
              "braycurtis",
              "jaccard",
              "matching",
              "dice",
              "kulsinski",
              "rogerstanimoto",
              "russellrao",
              "sokalmichener",
              "sokalsneath"],
        example="minkowski"
    ),
    'p': fields.Float(
        description="The power of the Minkowski metric to be used to calculate distance between points.",
        example=2,
    ),
}

# Gaussian mixture
definitions["clustering"]["gaussianmixture"] = {
    'n_components': fields.Integer(
        description="Number of components in the Gaussian mixture",
        example=3,
    ),
    'covariance_type': fields.String(
        description="Type of covariance free parameters for the Gaussians. Available options are:\n"
                    "- full: Each component has a free general covariance matrix\n"
                    "- tied: All of the component have the same general covariance matrix\n"
                    "- diag: Each component has a free diagonal covariance matrix\n"
                    "- spherical: Each component has a single free variance\n",
        example="full", enum=["full", "tied", "diag", "spherical"],
    ),
    'tol': fields.Float(
        description="Numerical tolerance for Expectation-Maximization calculations",
        example=1E-3,
    ),
    'max_iter': fields.Integer(
        description="Maximum number of iterations",
        example=100,
    ),
    'random_state': fields.Integer(
        description="Seed used for the random number generation for parameter initialization.",
        example=5
    ),
}

# Agglomerative
definitions["clustering"]["agglomerative"] = {
    'n_clusters': fields.Integer(
        description="Number of clusters to find",
        example=3,
    ),
    'linkage': fields.String(
        description="Linkage criterion to use, determining which distance to use between sets of observation. "
                    "Available options are:\n"
                    "- ward: minimize the variance of the clusters being merged. AFFINITY MUST BE SET TO euclidean\n"
                    "- average: uses the average of the distances of each observation of the two sets\n"
                    "- complete: uses the maximum distances between all observations of the two sets\n"
                    "- single: uses the minimum of the distances between all observations of the two sets\n",
        example="ward", enum=["ward", "average", "complete", "single"],
    ),
    'affinity': fields.String(
        description="Metric used to compute the linkage. "
                    "Available options are:\n"
                    "- euclidean: The Euclidean (l2) distance.\n"
                    "- manhattan: The Manhattan/Taxicab (l1) distance\n"
                    "- cosine: The cosine similarity (l2-normalized)\n",
        example="euclidean", enum=["euclidean", "manhattan", "cosine"],
    ),
}

# Spectral
definitions["clustering"]["spectral"] = {
    'n_clusters': fields.Integer(
        description="Number of clusters to find",
        example=3,
    ),
    'assign_labels': fields.String(
        description="Strategy used to assign the labels in the constructed embedding space. "
                    "Available options are:\n"
                    "- kmeans: a k-means clustering. Can match finer details, but it can be unstable. The random seed is important for reproducibility.\n"
                    "- discretize: tends to create parcels of fairly even and geometrical shape.\n",
        example="kmeans", enum=["kmeans", "discretize"],
    ),
    'random_state': fields.Integer(
        description="Seed used for the random number generation.",
        example=5
    ),
}
