# ## Getting started

using MLJ
import RDatasets: dataset
using PrettyPrinting
import DataFrames: DataFrame, select, Not

carseats = dataset("ISLR", "Carseats")

# We encode a new variable `High` based on whether the sales are higher or lower than 8 and add that column to the dataframe:

High = ifelse.(carseats.Sales .<= 8, "No", "Yes") |> x -> categorical(x; levels=["No", "Yes"], ordered=true)
carseats[!, :High] = High

train_validate, hold_out_test = partition(eachindex(High), 0.8, shuffle=true, rng=333)

# Let's now train a basic decision tree classifier for `High` given the other features after one-hot-encoding the categorical features:

train_validate_X = select(carseats, Not([:Sales, :High]))[train_validate, :]
train_validate_y = High[train_validate]

hold_out_test_X = select(carseats, Not([:Sales, :High]))[hold_out_test, :]
hold_out_test_y = High[hold_out_test]

# ### Decision Tree Classifier

DTC = @load DecisionTreeClassifier pkg = DecisionTree

scitype(train_validate_X)
HotTreeClf = OneHotEncoder() |> DTC()

mdl = HotTreeClf
mach = machine(mdl, train_validate_X, train_validate_y)
fit!(mach)

cv = StratifiedCV(nfolds=10; rng=112)
# performance_without_tuning = evaluate!(mach; resampling=cv, measures=[mcc, fpr, fnr, misclassification_rate], verbosity=0)

# Note `|>` is syntactic sugar for creating a `Pipeline` model from component model instances or model types.
# Note also that the machine `mach` is trained on the whole data.
ypred = predict_mode(mach, train_validate_X)
misclassification_rate(ypred, train_validate_y)
accuracy(ypred, train_validate_y)

# That's right... it gets it perfectly; this tends to be classic behaviour for a DTC to overfit the data it's trained on.
# Let's see if it generalises:

ypred = predict_mode(mach, hold_out_test_X)
perf_dt = misclassification_rate(ypred, hold_out_test_y)
mcc_dt = mcc(ypred, hold_out_test_y)

# ### Tuning a DTC
# Let's try to do a bit of tuning

r_mpi = range(mdl, :(decision_tree_classifier.max_depth), lower=1, upper=10)
r_msl = range(mdl, :(decision_tree_classifier.min_samples_leaf), lower=1, upper=20)

HotTreeClf = OneHotEncoder() |> DTC()
mdl = HotTreeClf

cv = StratifiedCV(nfolds=10; rng=112)
tm = TunedModel(
    model=mdl,
    ranges=[r_mpi, r_msl],
    tuning=Grid(resolution=10),
    resampling=cv,
    operation=predict_mode,
    measure=mcc,
)
mtm = machine(tm, train_validate_X, train_validate_y)
fit!(mtm)
# performance_upon_tuning = evaluate!(mtm, resampling=cv, measures=[mcc, fpr, fnr, misclassification_rate], verbosity=0)

ypred = predict_mode(mtm, hold_out_test_X)
perf_tuned_dt = misclassification_rate(ypred, hold_out_test_y)
mcc_tuned_dt = mcc(ypred, hold_out_test_y)

@show perf_dt, perf_tuned_dt
@show mcc_dt, mcc_tuned_dt

# We can inspect the parameters of the best model

fitted_params(mtm).best_model.decision_tree_classifier

@show models("CatBoost")

models(x -> x.is_supervised && x.is_pure_julia)  #lists all supervised models written in pure julia.

models(matching(train_validate_X))  #lists all unsupervised models compatible with input X.

models(matching(train_validate_X, train_validate_y))   # lists all supervised models compatible with input/target X/y.

models() do model
    matching(model, train_validate_X, train_validate_y) &&
        model.prediction_type == :probabilistic &&
        model.is_pure_julia
end

measures("f1")


# Using CatBoost

using CatBoost.MLJCatBoostInterface
CB = @load CatBoostClassifier pkg = CatBoost

# scitype(train_validate_X)
cb_mdl = OneHotEncoder() |> CB()
cb = machine(cb_mdl, train_validate_X, train_validate_y)
fit!(cb)

ypred = MLJ.predict_mode(cb, hold_out_test_X)
perf_cb = misclassification_rate(ypred, hold_out_test_y)
mcc_cb = mcc(ypred, hold_out_test_y)

@show perf_dt, perf_tuned_dt, perf_cb
@show mcc_dt, mcc_tuned_dt, mcc_cb