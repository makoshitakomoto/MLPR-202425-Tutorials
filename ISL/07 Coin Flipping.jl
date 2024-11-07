using Distributions
using Random
Random.seed!(12); # Set seed for reproducibility
using StatsPlots

### Make Synthetic Experimental data
p_true = 0.7
N = 100
data = rand(Bernoulli(p_true), N)

### Inference

using Turing

plot(Beta(1, 1), label = "Beta Dist")
plot!(Uniform(0, 1), label = "Uniform Dist")
# Unconditioned coinflip model with `N` observations.
@model function coinflip(; N::Int)
	# Our prior belief about the probability of heads in a coin toss.
	p ~ Uniform(0, 1)

	# Heads or tails of a coin are drawn from `N` independent and identically distributed Bernoulli distributions with success rate `p`.
	y ~ filldist(Bernoulli(p), N)
end


result_untrained = [rand(coinflip(; N)) for _ in 1:100]
@info "If the untrained coinflip experiment is run 100 times and then the p is avearged, then we get" mean([x.p for x in result_untrained])
# This is because we are using a P which is still the Prior, after training (this word does not really suit for Bayesian Inference but we use it to draw analogy with general ML) on the experimental data we will have the Posterior
using ReverseDiff

coinflip(y::AbstractVector{<:Real}) = coinflip(; N = length(y)) | (; y)
model = coinflip(data)
sampler = NUTS(; adtype = AutoReverseDiff(compile = true))

chain = sample(model, sampler, 1000, progress = false)
histogram(chain)

# Visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend).
mean_p = round(mean(chain[:p]); sigdigits = 3)
std_p = round(2 * std(chain[:p]); sigdigits = 3)
density(chain; xlim = (0, 1), legend = :left, w = 2, c = :blue, label = "Posterior $(mean_p) ± $(std_p)")


# Visualize the true probability of heads in red.
vline!([p_true]; label = "True probability $(p_true)", c = :red)
