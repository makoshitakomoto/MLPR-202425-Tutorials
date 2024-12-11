using Distributed
addprocs(8)
@everywhere begin
	using Distributions
	using FillArrays
	using StatsPlots
	using LinearAlgebra
	using Random
	using Turing
	using ReverseDiff

	### Generate Synthetic Training Data
	# Set a random seed.
	Random.seed!(3)

	# Define Gaussian mixture model.
	w = [0.5, 0.5]
	μ = [-3.5, 0.5]
	mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

	# We draw the data points.
	N = 60
	x = rand(mixturemodel, N)
	x .+= randn(size(x))
end
scatter(x[1, :], x[2, :]; legend = false, title = "Synthetic Dataset")
@everywhere x = $x

@everywhere @model function GMM(x, K) #Gaussian Mixture Model which takes x and K as arguments
	μ ~ Bijectors.ordered(MvNormal(Zeros(K), I))
	w ~ Dirichlet(K, 1.0)
	x ~ MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)
end
K = 3
model = GMM(x, K)
sampler = NUTS(; adtype = AutoReverseDiff(compile = true))
nsamples = 100
nchains = 8 #Repeat this example with 8 chains and show Distributed computing with Turing and Julia
ch = sample(model, sampler, MCMCDistributed(), nsamples, nchains)

# Model with mean of samples as parameters.
μ_mean = [mean(ch, "μ[$i]") for i in 1:3]
w_mean = [mean(ch, "w[$i]") for i in 1:3]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ_mean], w_mean)

using Flux: softmax
function sample_class(xi, dists, w)
	lvec = [(logpdf(d, xi) + log(w[i])) for (i, d) in enumerate(dists)]
	return argmax(softmax(lvec))
end

K = 3
D, N = size(x)

assignments = [sample_class(x[:, i], [MvNormal(Fill(μₖ, K), I) for μₖ in μ_mean], w_mean) for i in 1:N]

begin
	contour(range(-7.5, 3; length = 1_000), range(-6.5, 3; length = 1_000), (x, y) -> logpdf(mixturemodel_mean, [x, y]); levels = 10, fill = true)
	first_class = x[:, assignments.==1]
	second_class = x[:, assignments.==2]
	scatter!(first_class[1, :], first_class[2, :]; legend = true, label = "Class 1")
	scatter!(second_class[1, :], second_class[2, :]; legend = true, label = "Class 2")
end

#TODO for students - Play around and calculate the variances (scales or spreads) of the two blobs
