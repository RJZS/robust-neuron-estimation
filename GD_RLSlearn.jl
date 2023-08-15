using Plots, JLD
using ModelingToolkit, DifferentialEquations, LinearAlgebra, Distributions
using LaTeXStrings
# Random.seed!(123)
include("GD_odes.jl")

@variables t V(t) input(t) gKCa(t) gCaL(t) mNa(t) hNa(t) mKd(t) mAf(t) hAf(t) mAs(t) hAs(t) mCaL(t) mCaT(t) hCaT(t) mH(t) Ca(t)
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
@parameters gH=0. gCaT=0.5 # gKCa=25. gCaL=0.

new_err = true
new_noise = false
Tf = 70000
ramp_start = 50000

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
        T1 = ramp_start+8000 # When to switch noise
        noise = generate_input(-2, noise1_sf, noise2_sf, T1, stepsize,
                        filter1_coeff, filter2_coeff, 1)
        save("GD_noise_input.jld", "noise", noise, "stepsize", stepsize)
else
        data = load("GD_noise_input.jld")
        noise = data["noise"]
        stepsize = data["stepsize"]
end

# With constant current, threshold is between 1.5 and 2.
input_fn(t) = noise[Int(floor(t/stepsize))+1] # Zero-order hold
# input_fn(t) = 1500 < t < 2000 ? -5 : 2

function gCaL_fn(t)
    if t < ramp_start
        return 2.5
    elseif t < ramp_start+15000
        return 2.5 + 3*(t-ramp_start)/20000
    else
        return 4.75
    end
end
function gKCa_fn(t)
    if t < ramp_start
        return 5
    elseif t < ramp_start+15000 
        return 5 + 5.5*(t-ramp_start)/20000
    else
        return 9.125
    end
end
@register_symbolic input_fn(t)
@register_symbolic gKCa_fn(t)
@register_symbolic gCaL_fn(t)

τ_uncert = 0.04 # τ uncertainty. 0.1 corresponds to ±10%.
τ_errs = τ_uncert*(2*rand(12).-1) .+ 1
τ_uncert_true = 0.0
τ_errs_true = τ_uncert_true*(2*rand(12).-1) .+ 1

max_error = 4 # Max error in half-act [mV]
half_acts = max_error*(2*rand(12).-1) # 12th element is for KCa, not Ca as in tau errors.
max_err_true = 0
half_acts_true = max_err_true*(2*rand(12).-1)

gCaLCa = 3 # In Ca dynamics
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
    D(Ca) ~ (1/tau_Ca) * ((-αCa*gCaLCa*mCaL*(V-ECa))+(-β*gCaT*mCaT*hCaT*(V-ECa)) - Ca) 
],t)

# Modelling errors
# True values are (45, 60, 85) for (mCaL, mCaT, hCaT)
# The gates are 
# (mNa, hNa, mKd, mAf, hAf, mAs, hAs, 
# mCaL, mCaT, hCaT, mH). The true values are
# (25, 40, 15, 80, 60, 60, 20, 45, 60, 85, 85, -30)

k = 6 # Number of maximal conductances to learn
@variables Vh(t) (ϕ(t))[1:k] mNah(t) hNah(t) mKdh(t) mAfh(t) hAfh(t) mAsh(t) hAsh(t) mCaLh(t) mCaTh(t) hCaTh(t) mHh(t) Cah(t)
@variables (dθ(t))[1:k,1] (θ(t))[1:k,1] (Ψ(t))[1:k,1] (P(t))[1:k,1:k] varying_gain(t)
@parameters γ=8.

@named Identifier = ODESystem([
    ϕ[1] ~ -mNah*hNah*(V-ENa)/C,
    ϕ[2] ~ -mKdh*(V-EK)/C,
    ϕ[3] ~ -mCaLh*(V-ECa)/C,
    ϕ[4] ~ -mCaTh*hCaTh*(V-ECa)/C,
    ϕ[5] ~ -mKCainf(Cah-half_acts[12])*(V-EK)/C,
    ϕ[6] ~ -(V-Eleak)/C,

    varying_gain ~ (Ψ'*P*Ψ)[1],
    D(Vh) ~ dot(θ,ϕ) + γ*(1 + varying_gain)*(V-Vh) + input/C,
    input ~ input_fn(t),
    gCaL ~ gCaL_fn(t),
    gKCa ~ gKCa_fn(t),

    D(mNah) ~ (1/(τ_errs[1]*tau_mNa(V))) * (mNainf(V-half_acts[1]) - mNah),
    D(hNah) ~ (1/(τ_errs[2]*tau_hNa(V))) * (hNainf(V-half_acts[2]) - hNah),
    D(mKdh) ~ (1/(τ_errs[3]*tau_mKd(V))) * (mKdinf(V-half_acts[3]) - mKdh),
    D(mAfh) ~ (1/(τ_errs[4]*tau_mAf(V))) * (mAfinf(V-half_acts[4]) - mAfh),
    D(hAfh) ~ (1/(τ_errs[5]*tau_hAf(V))) * (hAfinf(V-half_acts[5]) - hAfh),
    D(mAsh) ~ (1/(τ_errs[6]*tau_mAs(V))) * (mAsinf(V-half_acts[6]) - mAsh),
    D(hAsh) ~ (1/(τ_errs[7]*tau_hAs(V))) * (hAsinf(V-half_acts[7]) - hAsh),
    D(mCaLh) ~ (1/(τ_errs[8]*tau_mCaL(V))) * (mCaLinf(V-half_acts[8]) - mCaLh),
    D(mCaTh) ~ (1/(τ_errs[9]*tau_mCaT(V))) * (mCaTinf(V-half_acts[9]) - mCaTh),
    D(hCaTh) ~ (1/(τ_errs[10]*tau_hCaT(V))) * (hCaTinf(V-half_acts[10]) - hCaTh),
    D(mHh) ~ (1/(τ_errs[11]*tau_mH(V))) * (mHinf(V-half_acts[11]) - mHh),
    D(Cah) ~ (1/(τ_errs[12]*tau_Ca)) * ((-αCa*gCaLCa*mCaLh*(V-ECa))+(-β*gCaT*mCaTh*hCaTh*(V-ECa)) - Cah),
],t)

@parameters α=0.005
@variables (PΨ(t))[1:k,1] (outer(t))[1:k,1:k]
@named UpdateLaw = ODESystem([
        [PΨ[i] ~ (P*Ψ)[i] for i=1:k]...,
        [D(θ[i]) ~ γ * PΨ[i] * (V-Vh) for i=1:k]...,
        #[D(θ[i]) ~ IfElse.ifelse(θ[i] <= 0, max(dθ[i],0), dθ[i]) for i=1:k]...,
        [D(Ψ[i])  ~ -γ*Ψ[i] + ϕ[i] for i=1:k]... 
],t)

@named dP = ODESystem(D.(P) ~ α*P - γ* P*Ψ * Ψ'*P)

full_sys = compose(ODESystem([
    Identifier.V ~ NeurTrue.V,
    [UpdateLaw.θ[i] ~ Identifier.θ[i] for i=1:k]...,
    [UpdateLaw.ϕ[i] ~ Identifier.ϕ[i] for i=1:k]...,
    [UpdateLaw.Ψ[i] ~ Identifier.Ψ[i] for i=1:k]...,
    [dP.Ψ[i] ~ Identifier.Ψ[i] for i=1:k]...,
    [UpdateLaw.P[i,j] ~ Identifier.P[i,j] for i=1:k for j=1:k]...,
    [dP.P[i,j] ~ Identifier.P[i,j] for i=1:k for j=1:k]...,
    UpdateLaw.V ~ NeurTrue.V,
    UpdateLaw.Vh ~ Identifier.Vh,
], t; name=:full_sys), [NeurTrue, Identifier, UpdateLaw, dP])

sys = structural_simplify(full_sys)

paramtrs = []

V0 = -80.
init_state_vec = [NeurTrue.V => V0, NeurTrue.mNa => mNainf(V0), NeurTrue.hNa => hNainf(V0), NeurTrue.mKd => mKdinf(V0),
    NeurTrue.mAf => 0., NeurTrue.hAf => 0., NeurTrue.mAs=>0.,NeurTrue.hAs=>0.,NeurTrue.mCaL=>mCaLinf(V0),
    NeurTrue.mCaT=>0.,NeurTrue.hCaT=>0.,NeurTrue.mH=>0.,NeurTrue.Ca=>(-αCa*4*mCaLinf(V0)*(V0-ECa))+(-β*0*0*0*(V0-ECa)),
    Identifier.Vh => 0., Identifier.mNah => 0., Identifier.hNah => 0., Identifier.mKdh => 0.,
    Identifier.mAfh => 0., Identifier.hAfh => 0., Identifier.mAsh=>0.,Identifier.hAsh=>0.,Identifier.mCaLh=>0.,
    Identifier.mCaTh=>0.,Identifier.hCaTh=>0.,Identifier.mHh=>0.,Identifier.Cah=>0.,
    [UpdateLaw.θ[i] => 10.0 for i=1:k]...,
    [UpdateLaw.Ψ[i] => 10.0 for i=1:k]...,
    [Identifier.ϕ[i] => 0.0 for i=1:k]...,
    [dP.P[i,j] => (i==j ? 1.0 : 0.0) for i=1:k for j=1:k]...,]
    
prob = ODEProblem(sys, init_state_vec, (0.,Tf), paramtrs)

dt = 0.1
sol = solve(prob, saveat=dt, alg_hints=[:stiff]); 

# Calculate error
err_trunc_st = Int((ramp_start-4000)/dt) # Covering gCaT ramp and 4k ms on either side
err_trnc_end = Int((ramp_start+20000)/dt)
error_signal = sol[NeurTrue.V,err_trunc_st:err_trnc_end].-sol[Identifier.Vh,err_trunc_st:err_trnc_end]
error_rms = sqrt(dot(error_signal,error_signal))

println("Error: $error_rms")
global current_err = error_rms

is_plotting = false
if is_plotting
    pVonlytrnc= plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.V,err_trunc_st:err_trnc_end],label=L"$v$",ylabel="[mV]",legend=:bottomleft,dpi=300)
    zoomed_start = Int((ramp_start-4000)/dt); zoomed_end = Int((ramp_start+10000)/dt)
    pVonlyzoomed= plot(sol.t[zoomed_start:zoomed_end], sol[NeurTrue.V,zoomed_start:zoomed_end],label=L"$v$",legend=:bottomright)
    pVtrnc= plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.V,err_trunc_st:err_trnc_end],label=L"$v$",legend=:bottomleft,ylabel="[mV]",dpi=300,size=1.2 .* (600,450))
    plot!(sol.t[err_trunc_st:err_trnc_end],sol[Identifier.Vh,err_trunc_st:err_trnc_end],label=L"$\hat{v}$",legend=:bottomleft,dpi=300)
    #plot!(sol.t[err_trunc_st:err_trnc_end],sol[NeurTrue.input,err_trunc_st:err_trnc_end])
    perrtrnc = plot(sol.t[err_trunc_st:err_trnc_end],
        abs.(sol[NeurTrue.V,err_trunc_st:err_trnc_end].-sol[Identifier.Vh,err_trunc_st:err_trnc_end]),label=L"|v-\hat{v}\,|",legend=:bottomleft,ylabel="[mV]",dpi=300,size=1.2 .* (600,450))

    pinputtrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.input,err_trunc_st:err_trnc_end],label=L"$u$",ylabel=L"[$\mu$A]",legend=:bottomleft,dpi=300)
    pgCaLtrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gCaL,err_trunc_st:err_trnc_end],label=L"$\mu_{\rm{CaL}}$",legend=:bottomright,xlabel="t [ms]",ylabel="[mS / cm^2]")
    plot!(sol.t[err_trunc_st:err_trnc_end], sol[Identifier.θ[3],err_trunc_st:err_trnc_end],label=false)
    pgKCatrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gKCa,err_trunc_st:err_trnc_end],label=L"$\mu_{\rm{KCa}}$",legend=:bottomright,xlabel="t [ms]",ylabel="[mS / cm^2]")
    plot!(sol.t[err_trunc_st:err_trnc_end], sol[Identifier.θ[5],err_trunc_st:err_trnc_end],label=false)
    pramps = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gCaL,err_trunc_st:err_trnc_end],label=L"$\mu_{\rm{CaL}}$",legend=:topleft,xlabel="t [ms]",ylabel="[mS / cm^2]",dpi=300,size=1.2 .* (600,450))
    plot!(sol.t[err_trunc_st:err_trnc_end], sol[Identifier.θ[3],err_trunc_st:err_trnc_end],label=L"$\hat{\theta}_{\rm{CaL}}$",dpi=300)
    plot!(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gKCa,err_trunc_st:err_trnc_end],label=L"$\mu_{\rm{KCa}}$",legend=:topleft,dpi=300)
    plot!(sol.t[err_trunc_st:err_trnc_end], sol[Identifier.θ[5],err_trunc_st:err_trnc_end],label=L"$\hat{\theta}_{\rm{KCa}}$",dpi=300)
    fig1 = plot(pVonlytrnc, pinputtrnc, pramps, layout=(3,1))
    savefig(fig1,"fig1_task")
    fig2 = plot(pVtrnc,perrtrnc,pramps,layout=(3,1))
    # savefig(fig2,"fig2_noerr")
    savefig(fig2,"fig3_centralised")

    pV=plot(sol, idxs=[NeurTrue.V, NeurTrue.input], legend=:bottomright)
    pVh=plot(sol,idxs=[Identifier.Vh])
    # plot!(sol, idxs=[Identifier.Vh])
    pgCaT = plot(sol, idxs=[NeurTrue.gCaT], width= 2)

    j=length(sol.t)
    perr = plot(sol.t[1:j], sol[NeurTrue.V,1:j] .- sol[Identifier.Vh,1:j])

    ptest=plot(sol, idxs=[dP.P[1,1]])

    p1=plot(sol, idxs=[Identifier.θ[1]], label=false)#,ylims=(0,0.01))
    hline!([100], width=1.5, label="gNa")
    p2=plot(sol, idxs=[Identifier.θ[2]], label=false)
    hline!([65], width=1.5, label="gKd")
    p3=plot(sol, idxs=[Identifier.θ[3]], label=false)
    plot!(sol.t, gCaL_fn, width=1.5, label="gCaL")
    p4=plot(sol, idxs=[Identifier.θ[4]], label=false)
    hline!([0.5], width=1.5, label="gCaT")
    p5=plot(sol, idxs=[Identifier.θ[5]], label=false)
    plot!(sol.t, gKCa_fn, width=1.5, label="gKCa")
    p6=plot(sol, idxs=[Identifier.θ[6]], label=false)
    hline!([0.3], width=1.5, label="gleak")

    perrz = plot(sol.t[i:j], sol[NeurTrue.V,i:j] .- sol[Identifier.Vh,i:j])

    pB = plot(p4,p5,layout=(2,1))
    pA = plot(p1,p2,p3,layout=(5,1))

    pC = plot(pV, pgCaT,layout=(2,1))
    pCtrnc = plot(pVonlytrnc, pgCaTtrnc,layout=(2,1))
    savefig(p)

    pV
end