# ## Getting started

using MLJ
import RDatasets: dataset
using PrettyPrinting
import DataFrames: DataFrame, select, Not

# ### Decision Tree Regressor

DTR = @load DecisionTreeRegressor pkg = DecisionTree

boston = dataset("MASS", "Boston")

y, X = unpack(boston, ==(:MedV))

train_validate, hold_out_test = partition(eachindex(y), 0.9, shuffle=true, rng=333)

# Let's now train a basic decision tree classifier for `High` given the other features after one-hot-encoding the categorical features:
begin
    train_validate_X = X[train_validate, :]
    train_validate_y = y[train_validate]

    hold_out_test_X = X[hold_out_test, :]
    hold_out_test_y = y[hold_out_test]

    scitype(train_validate_X)

    # Let's recode the Count as Continuous and then fit a DTR

    train_validate_X = coerce(train_validate_X, autotype(train_validate_X, rules=(:discrete_to_continuous,)))
    scitype(train_validate_X)

    dtr_model = DTR()
    dtr = machine(dtr_model, train_validate_X, train_validate_y)
    fit!(dtr)

    ypred = MLJ.predict(dtr, hold_out_test_X)
    perf_dt = round(rms(ypred, hold_out_test_y), sigdigits=3)

    # Again we can try tuning this a bit, since it's the same idea as before, let's just try to adjust the depth of the tree:

    r_depth = range(dtr_model, :max_depth, lower=-2, upper=20)
    r_msl = range(dtr_model, :min_samples_leaf, lower=1, upper=50)

    cv = CV(nfolds=10; rng=1555)
    tm = TunedModel(
        model=dtr_model,
        ranges=[r_depth, r_msl],
        tuning=Grid(resolution=10),
        resampling=cv,
        measure=rms,
    )
    mtm = machine(tm, train_validate_X, train_validate_y)
    fit!(mtm)

    ypred = MLJ.predict(mtm, hold_out_test_X)
    perf_tuned_dt = round(rms(ypred, hold_out_test_y), sigdigits=3)

    rep = report(mtm)
    rep.best_model

    # ## Random Forest
    #
    # **Note**: the package [`DecisionTree.jl`](https://github.com/bensadeghi/DecisionTree.jl) also has a RandomForest model but it is not yet interfaced with in MLJ.

    RFR = @load RandomForestRegressor pkg = MLJScikitLearnInterface

    rf_mdl = RFR()
    rf = machine(rf_mdl, train_validate_X, train_validate_y)
    fit!(rf)

    ypred = MLJ.predict(rf, hold_out_test_X)
    perf_rf = round(rms(ypred, hold_out_test_y), sigdigits=3)


    # ## Gradient Boosting Machine

    XGBR = @load XGBoostRegressor

    xgb_mdl = XGBR()
    xgb = machine(xgb_mdl, train_validate_X, train_validate_y)
    fit!(xgb)

    ypred = predict(xgb, hold_out_test_X)
    perf_xgb = round(rms(ypred, hold_out_test_y), sigdigits=3)

    @show perf_dt, perf_tuned_dt, perf_rf, perf_xgb

end
# Again we could do some tuning for this.

r_1 = range(xgb_mdl, :max_depth, lower=1, upper=10)
r_2 = range(xgb_mdl, :min_child_weight, lower=1, upper=6)

xgb_tuned = TunedModel(
    model=xgb_mdl,
    ranges=[r_1, r_2],
    tuning=Grid(resolution=10),
    resampling=cv,
    measure=rms,
)
mtm = machine(xgb_tuned, train_validate_X, train_validate_y)
fit!(mtm)

ypred = MLJ.predict(mtm, hold_out_test_X)
perf_tuned_xgb = round(rms(ypred, hold_out_test_y), sigdigits=3)

rep = report(mtm)
rep.best_model

@show perf_dt, perf_tuned_dt, perf_rf, perf_xgb, perf_tuned_xgb
