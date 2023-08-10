using Plots, JLD
using ModelingToolkit, DifferentialEquations, LinearAlgebra, Distributions
using LaTeXStrings
# Random.seed!(123)
include("GD_odes.jl")

@variables t V(t) input(t) gCaT(t) gKCa(t) gCaL(t) mNa(t) hNa(t) mKd(t) mAf(t) hAf(t) mAs(t) hAs(t) mCaL(t) mCaT(t) hCaT(t) mH(t) Ca(t)
D = Differential(t)

## Constant simulation parameters

## Definition of reversal potential values. 
const ENa = 40.; # Sodium reversal potential
const EK = -90.; # Potassium reversal potential
const ECa = 120.; # Calcium reversal potential
const EH= -40.; # Reversal potential for the H-current (permeable to both sodium and potassium ions)
const Eleak = -50.; # Reversal potential of leak channels

const C=0.1; # Membrane capacitance
αCa=0.1; # Calcium dynamics (L-current)
β=0.06 # Calcium dynamics (T-current) # Default is 0.1

# Maximal conductances for the leak, sodium, delayed-rectifier potassium,
# fast A-type potassium, slow A-type potassium, calcium-activated potassium,
# L-type calcium, T-type calcium and H currents.
@parameters gl=0.3 gNa=100. gKd=65. gAf=0. gAs=0.
@parameters gH=0. # gKCa=25. gCaL=0. # gCaT=3.0

new_noise = false
Tf = 30000

if new_noise
    # Noise-generated current, two parts
        function generate_input(Iconst, noise1_sf, noise2_sf, T1, stepsize, filter1_coeff, filter2_coeff, num_inputs)
            d = Normal(0,1)
            noise1 = rand(d, (Int(T1/(stepsize))+1,num_inputs))*noise1_sf
            for i in eachindex(noise1)
                i == 1 ? noise1[i] = 0 : noise1[i]=noise1[i-1]+filter1_coeff*(noise1[i]-noise1[i-1])
            end
            noise2 = rand(d, (Int((Tf-T1)/(stepsize)),num_inputs))*noise2_sf
            for i in eachindex(noise2)
                i == 1 ? noise2[i] = 0 : noise2[i]=noise2[i-1]+filter2_coeff*(noise2[i]-noise2[i-1])
            end
            noise = [noise1... noise2...] .+ Iconst
        end
        stepsize = 1
        noise1_sf = 1.4; noise2_sf = 7
        filter1_coeff = 0.1; filter2_coeff = 0.01
        T1 = 15000 # When to switch noise
        noise = generate_input(-2, noise1_sf, noise2_sf, T1, stepsize,
                        filter1_coeff, filter2_coeff, 1)
        save("GD_noise_input.jld", "noise", noise, "stepsize", stepsize)
    else
        data = load("GD_noise_input.jld")
        noise = data["noise"]
        stepsize = data["stepsize"]
    end

# input_fn(t) = -0.5 # -1.5
input_fn(t) = noise[Int(floor(t/stepsize))+1] # Zero-order hold

function gCaT_ramp(t)
    if t < ramp_start
        return 3
    elseif ramp_start <= t < ramp_start+16000
        return 0.0001875*(t-ramp_start) + 3
    else
        return 6
    end
end
function gCaT_fn(t)
    return 0.5
end
function gCaL_fn(t)
    if t < 5000
        return 2.5
    elseif t < 20000
        return 2.5 + 3*(t-5000)/20000
    else
        return 4.75
    end
end
function gKCa_fn(t)
    if t < 5000
        return 5
    elseif t < 20000 
        return 5 + 5.5*(t-5000)/20000
    else
        return 9.125
    end
end
@register_symbolic input_fn(t)
@register_symbolic gCaT_fn(t)
@register_symbolic gKCa_fn(t)
@register_symbolic gCaL_fn(t)

τ_uncert = 0.0 # τ uncertainty. 0.1 corresponds to ±10%.
τ_errs = τ_uncert*(2*rand(12).-1) .+ 1
τ_uncert_true = 0.0
τ_errs_true = τ_uncert_true*(2*rand(12).-1) .+ 1

max_error = 0 # Max error in half-act [mV]
half_acts = max_error*(2*rand(12).-1) # 12th element is for KCa, not Ca as in tau errors.
max_err_true = 0
half_acts_true = max_err_true*(2*rand(12).-1)

@named NeurTrue = ODESystem([
    D(V) ~ (1/C) * (-gNa*mNa*hNa*(V-ENa) +
    # Potassium Currents
    -gKd*mKd*(V-EK) -gAf*mAf*hAf*(V-EK) -gAs*mAs*hAs*(V-EK) +
    -gKCa*mKCainf(Ca-half_acts_true[12])*(V-EK) +
    # Calcium currents
    -gCaL*mCaL*(V-ECa) +
    -gCaT*mCaT*hCaT*(V-ECa) +
    # Cation current
    -gH*mH*(V-EH) +
    # Passive currents
    -gl*(V-Eleak) +
    # Stimulation currents
    # +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))
    +input),
    input ~ input_fn(t),
    gCaT ~ gCaT_fn(t),
    gCaL ~ gCaL_fn(t),
    gKCa ~ gKCa_fn(t),
    D(mNa) ~ (1/(τ_errs_true[1]*tau_mNa(V))) * (mNainf(V-half_acts_true[1]) - mNa),
    D(hNa) ~ (1/(τ_errs_true[2]*tau_hNa(V))) * (hNainf(V-half_acts_true[2]) - hNa),
    D(mKd) ~ (1/(τ_errs_true[3]*tau_mKd(V))) * (mKdinf(V-half_acts_true[3]) - mKd),
    D(mAf) ~ (1/(τ_errs_true[4]*tau_mAf(V))) * (mAfinf(V-half_acts_true[4]) - mAf),
    D(hAf) ~ (1/(τ_errs_true[5]*tau_hAf(V))) * (hAfinf(V-half_acts_true[5]) - hAf),
    D(mAs) ~ (1/(τ_errs_true[6]*tau_mAs(V))) * (mAsinf(V-half_acts_true[6]) - mAs),
    D(hAs) ~ (1/(τ_errs_true[7]*tau_hAs(V))) * (hAsinf(V-half_acts_true[7]) - hAs),
    D(mCaL) ~ (1/(τ_errs_true[8]*tau_mCaL(V))) * (mCaLinf(V-half_acts_true[8]) - mCaL),
    D(mCaT) ~ (1/(τ_errs_true[9]*tau_mCaT(V))) * (mCaTinf(V-half_acts_true[9]) - mCaT),
    D(hCaT) ~ (1/(τ_errs_true[10]*tau_hCaT(V))) * (hCaTinf(V-half_acts_true[10]) - hCaT),
    D(mH) ~ (1/(τ_errs_true[11]*tau_mH(V))) * (mHinf(V-half_acts_true[11]) - mH),
    D(Ca) ~ (1/(τ_errs_true[12]*tau_Ca)) * ((-αCa*gCaL*mCaL*(V-ECa))+(-β*2.5*mCaT*hCaT*(V-ECa)) - Ca) 
],t)

# Modelling errors
# True values are (45, 60, 85) for (mCaL, mCaT, hCaT)
# The gates are 
# (mNa, hNa, mKd, mAf, hAf, mAs, hAs, 
# mCaL, mCaT, hCaT, mH). The true values are
# (25, 40, 15, 80, 60, 60, 20, 45, 60, 85, 85, -30)


sys = structural_simplify(NeurTrue)

paramtrs = []

V0 = -80.
init_state_vec = [V => V0, mNa => mNainf(V0), hNa => hNainf(V0), mKd => mKdinf(V0),
    mAf => 0., hAf => 0., mAs=>0.,hAs=>0.,mCaL=>mCaLinf(V0),
    mCaT=>0.,hCaT=>0.,mH=>0.,Ca=>(-αCa*4*mCaLinf(V0)*(V0-ECa))+(-β*0*0*0*(V0-ECa)),
    Identifier.Vh => 0., Identifier.mNah => 0., Identifier.hNah => 0., Identifier.mKdh => 0.,
    Identifier.mAfh => 0., Identifier.hAfh => 0., Identifier.mAsh=>0.,Identifier.hAsh=>0.,Identifier.mCaLh=>0.,
    Identifier.mCaTh=>0.,Identifier.hCaTh=>0.,Identifier.mHh=>0.,Identifier.Cah=>0.,
    [UpdateLaw.θ[i] => 10.0 for i=1:k]...,
    [UpdateLaw.Ψ[i] => 10.0 for i=1:k]...,
    [Identifier.ϕ[i] => 0.0 for i=1:k]...,
    [dP.P[i,j] => (i==j ? 1.0 : 0.0) for i=1:k for j=1:k]...,]
    
prob = ODEProblem(sys, init_state_vec, (0.,Tf), paramtrs)

dt = 0.1
sol = solve(prob, saveat=dt); 

p1=plot(sol, idxs=[NeurTrue.V],legend=false,dpi=600)
p2=plot(sol, idxs=[NeurTrue.input],legend=false)
p3=plot(sol,idxs=[NeurTrue.gKCa,NeurTrue.gCaL,NeurTrue.gCaT])
plot(p1,p2,p3,layout=(3,1))

p1