using ReinforcementLearning
# using AlphaZero (ConnectFour) # GOOGLE - Master of GO, Chess etc. Also AlphaFold for protein folding (Nobel Prize)
# using MuZero (TicToe) (github.com/deveshjawla) # Google - Master of Atari games, other simulation based games whcih require vision

env = TicTacToeEnv()
S = state_space(env)
s = state(env)
A = action_space(env)
is_terminated(env)
while true
	RLBase.act!(env, rand(A))
	is_terminated(env) && break
end
state(env)
reward(env)

#In this simple game, we are interested in finding out an optimum policy for the agent to gain the maximum cumulative reward in an episode. 

#The random selection policy above is a good benchmark. The only thing left is to calculate the total reward.
run(
	RandomPolicy(),
	RandomWalk1D(),
	StopAfterNEpisodes(10),
	TotalRewardPerEpisode(),
)

#Next, let's introduce one of the most common policies, the QBasedPolicy
NS = length(S)
NA = length(A)
policy = QBasedPolicy(
	learner = TDLearner(
		TabularQApproximator(
			n_state = NS,
			n_action = NA,
		),
		:SARS,
	),
	explorer = EpsilonGreedyExplorer(0.1), #Here we choose the TDLearner and the EpsilonGreedyExplorer. But you can also replace them with some other Q value learners or value explorers. 
)

run(
	policy,
	RandomWalk1D(),
	StopAfterNEpisodes(10),
	TotalRewardPerEpisode(),
)

#optimise!()

# A trajectory (also called Experience Replay Buffer in reinforcement learning literature)
trajectory = Trajectory(
	ElasticArraySARTSTraces(;
		state = Int64 => (),
		action = Int64 => (),
		reward = Float64 => (),
		terminal = Bool => (),
	),
	EpisodesSampler(),
	InsertSampleRatioController(),
)

agent = Agent(
	policy = RandomPolicy(),
	trajectory = trajectory,
)

run(agent, env, StopAfterNEpisodes(100), TotalRewardPerEpisode(; is_display_on_exit = true))

optimise!(policy, trajectory)