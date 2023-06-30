global num_dps = 10

global current_err = 0.
global all_errs = zeros(num_dps)

for i=1:num_dps
    println("Trial $i")

    include("GD_RLSlearn.jl")
    # include("GD_RLSlearndist.jl")
    # include("GD_RLSlearndist_redundant.jl")

    all_errs[i] = current_err
end

mean_errs = mean(all_errs)
using Statistics; errs_std = std(all_errs) # Standard deviation

println("Mean: $mean_errs")
println("Std: $errs_std")