using Plots, JLD
using ModelingToolkit, DifferentialEquations, LinearAlgebra, Distributions, LaTeXStrings
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

@named NeurTrue = ODESystem([
    D(V) ~ (1/C) * (-gNa*mNa*hNa*(V-ENa) +
    # Potassium Currents
    -gKd*mKd*(V-EK) -gAf*mAf*hAf*(V-EK) -gAs*mAs*hAs*(V-EK) +
    -gKCa*mKCainf(Ca)*(V-EK) +
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
    D(mNa) ~ (1/tau_mNa(V)) * (mNainf(V) - mNa),
    D(hNa) ~ (1/tau_hNa(V)) * (hNainf(V) - hNa),
    D(mKd) ~ (1/tau_mKd(V)) * (mKdinf(V) - mKd),
    D(mAf) ~ (1/tau_mAf(V)) * (mAfinf(V) - mAf),
    D(hAf) ~ (1/tau_hAf(V)) * (hAfinf(V) - hAf),
    D(mAs) ~ (1/tau_mAs(V)) * (mAsinf(V) - mAs),
    D(hAs) ~ (1/tau_hAs(V)) * (hAsinf(V) - hAs),
    D(mCaL) ~ (1/tau_mCaL(V)) * (mCaLinf(V) - mCaL),
    D(mCaT) ~ (1/tau_mCaT(V)) * (mCaTinf(V) - mCaT),
    D(hCaT) ~ (1/tau_hCaT(V)) * (hCaTinf(V) - hCaT),
    D(mH) ~ (1/tau_mH(V)) * (mHinf(V) - mH),
    D(Ca) ~ (1/tau_Ca) * ((-αCa*gCaL*mCaL*(V-ECa))+(-β*2.5*mCaT*hCaT*(V-ECa)) - Ca) 
],t)

# Modelling errors
# True values are (45, 60, 85) for (mCaL, mCaT, hCaT)
# The gates are 
# (mNa, hNa, mKd, mAf, hAf, mAs, hAs, 
# mCaL, mCaT, hCaT, mH). The true values are
# (25, 40, 15, 80, 60, 60, 20, 45, 60, 85, 85, -30)

l = 3 # Level of redundancy

τ_uncert = 0.04 # τ uncertainty. 0.1 corresponds to ±10%.
τ_errs = τ_uncert*(2*rand(14*l).-1) .+ 1

max_error = 4 # Max error in half-act [mV]
half_acts = max_error*(2*rand(14*l).-1) # 14th element is for KCa, not Ca as in tau errors.

k = 6 # Number of maximal conductances to learn
q = (k-1)*l+1 # Length of phi vector (no redundancy for leak current)
@variables Vh(t) (ϕ(t))[1:q] (mNah(t))[1:l,1] (hNah(t))[1:l,1] (mKdh(t))[1:l,1] (mCaLh(t))[1:l,1] (mCaTh(t))[1:l,1] (hCaTh(t))[1:l,1] (Cah(t))[1:l,1]
@variables (dθ(t))[1:q,1] (θ(t))[1:q,1] (Ψ(t))[1:q,1] (P(t))[1:q,1] (tmp(t))[1:q,1]
@parameters γ0=8
γ = 8*ones(q)

@named Identifier = ODESystem([
    [ϕ[i] ~ -mNah[i]*hNah[i]*(V-ENa)/C for i=1:l]...,
    [ϕ[l+i] ~ -mKdh[i]*(V-EK)/C for i=1:l]...,
    [ϕ[2*l+i] ~ -mCaLh[i]*(V-ECa)/C for i=1:l]...,
    [ϕ[3*l+i] ~ -mCaTh[i]*hCaTh[i]*(V-ECa)/C for i=1:l]...,
    [ϕ[4*l+i] ~ -mKCainf(Cah[i]-half_acts[6*l+i])*(V-EK)/C for i=1:l]...,
    ϕ[5*l+1] ~ -(V-Eleak)/C,

    [tmp[i] ~ γ[i]*P[i]*Ψ[i]^2 for i=1:q]...,
    D(Vh) ~ dot(θ,ϕ) + (γ0+sum(tmp))*(V-Vh) + input/C,
    input ~ input_fn(t),

    [D(mNah[i]) ~ (1/(τ_errs[i]*tau_mNa(V))) * (mNainf(V-half_acts[i]) - mNah[i]) for i=1:l]...,
    [D(hNah[i]) ~ (1/(τ_errs[l+i]*tau_hNa(V))) * (hNainf(V-half_acts[l+i]) - hNah[i]) for i=1:l]...,
    [D(mCaLh[i]) ~ (1/(τ_errs[2*l+i]*tau_mCaL(V))) * (mCaLinf(V-half_acts[2*l+i]) - mCaLh[i]) for i=1:l]...,
    [D(mCaTh[i]) ~ (1/(τ_errs[3*l+i]*tau_mCaT(V))) * (mCaTinf(V-half_acts[3*l+i]) - mCaTh[i]) for i=1:l]...,
    [D(hCaTh[i]) ~ (1/(τ_errs[4*l+i]*tau_hCaT(V))) * (hCaTinf(V-half_acts[4*l+i]) - hCaTh[i]) for i=1:l]...,
    [D(mKdh[i]) ~ (1/(τ_errs[5*l+i]*tau_mKd(V))) * (mKdinf(V-half_acts[5*l+i]) - mKdh[i]) for i=1:l]...,
    [D(Cah[i]) ~ (1/(τ_errs[6*l+i]*tau_Ca)) * ((-β*2.5*mCaTh[i]*hCaTh[i]*(V-ECa)) - Cah[i]) for i=1:l]...,
],t)

@variables meanNa(t) meanKd(t) meanKCa(t) meanCaT(t) meanCaL(t)
@parameters α=0.0002 beta=5e-5
@named UpdateLaw = ODESystem([
        meanNa ~ sum(θ[1:l])/l,
        meanKd ~ sum(θ[l+1:2*l])/l,
        meanCaL ~ sum(θ[2*l+1:3*l])/l,
        meanCaT ~ sum(θ[3*l+1:4*l])/l,
        meanKCa ~ sum(θ[4*l+1:5*l])/l,

        [D(θ[i]) ~ γ[i] * P[i] * Ψ[i] * (V-Vh) - beta*(θ[i]-meanNa) for i=1:l]...,
        [D(θ[i]) ~ γ[i] * P[i] * Ψ[i] * (V-Vh) - beta*(θ[i]-meanKd) for i=l+1:2*l]...,
        [D(θ[i]) ~ γ[i] * P[i] * Ψ[i] * (V-Vh) - beta*(θ[i]-meanCaL) for i=2*l+1:3*l]...,
        [D(θ[i]) ~ γ[i] * P[i] * Ψ[i] * (V-Vh) - beta*(θ[i]-meanCaT) for i=3*l+1:4*l]...,
        [D(θ[i]) ~ γ[i] * P[i] * Ψ[i] * (V-Vh) - beta*(θ[i]-meanKCa) for i=4*l+1:5*l]...,
        D(θ[5*l+1]) ~ γ[5*l+1] * P[5*l+1] * Ψ[5*l+1] * (V-Vh),
        #[D(θ[i]) ~ IfElse.ifelse(θ[i] <= 0, max(dθ[i],0), dθ[i]) for i=1:k]...,
        [D(Ψ[i])  ~ -γ[i]*Ψ[i] + ϕ[i] for i=1:q]... 
],t)

@named dP = ODESystem([
    [D(P[i]) ~ (α*P[i] - α*P[i]*Ψ[i] * Ψ[i]*P[i]) for i=1:q]...
],t)

full_sys = compose(ODESystem([
    Identifier.V ~ NeurTrue.V,
    [UpdateLaw.θ[i] ~ Identifier.θ[i] for i=1:q]...,
    [UpdateLaw.ϕ[i] ~ Identifier.ϕ[i] for i=1:q]...,
    [UpdateLaw.Ψ[i] ~ Identifier.Ψ[i] for i=1:q]...,
    [dP.Ψ[i] ~ Identifier.Ψ[i] for i=1:q]...,
    [UpdateLaw.P[i] ~ Identifier.P[i] for i=1:q]...,
    [dP.P[i] ~ Identifier.P[i] for i=1:q]...,
    UpdateLaw.V ~ NeurTrue.V,
    UpdateLaw.Vh ~ Identifier.Vh,
    #Identifier.Ca ~ NeurTrue.Ca,
], t; name=:full_sys), [NeurTrue, Identifier, UpdateLaw, dP])

sys = structural_simplify(full_sys)

paramtrs = []

V0 = -80.
init_state_vec = [NeurTrue.V => V0, NeurTrue.mNa => mNainf(V0), NeurTrue.hNa => hNainf(V0), NeurTrue.mKd => mKdinf(V0),
    NeurTrue.mAf => 0., NeurTrue.hAf => 0., NeurTrue.mAs=>0.,NeurTrue.hAs=>0.,NeurTrue.mCaL=>mCaLinf(V0),
    NeurTrue.mCaT=>0.,NeurTrue.hCaT=>0.,NeurTrue.mH=>0.,NeurTrue.Ca=>(-αCa*4*mCaLinf(V0)*(V0-ECa))+(-β*0*0*0*(V0-ECa)),
    Identifier.Vh => 0., [Identifier.mNah[i] => 0. for i=1:l]..., [Identifier.hNah[i] => 0. for i=1:l]..., [Identifier.mKdh[i] => 0. for i=1:l]...,
    [Identifier.mCaTh[i] => 0. for i=1:l]...,[Identifier.hCaTh[i] => 0. for i=1:l]...,[Identifier.mCaLh[i] => 0. for i=1:l]...,[Identifier.Cah[i] => 0. for i=1:l]...,
    [UpdateLaw.θ[i] => 10.0 for i=1:q]...,
    [UpdateLaw.Ψ[i] => 0.0 for i=1:q]...,
    [Identifier.ϕ[i] => 0.0 for i=1:q]...,
    [dP.P[i] => 1.0 for i=1:q]...,
    ]
    
prob = ODEProblem(sys, init_state_vec, (0.,Tf), paramtrs)

dt = 0.1
sol = solve(prob, saveat=dt)#, alg_hints=[:stiff]); 

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
    pVtrnc= plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.V,err_trunc_st:err_trnc_end],label=L"$v$",legend=:bottomleft,ylabel="[mV]",dpi=300)
    plot!(sol.t[err_trunc_st:err_trnc_end],sol[Identifier.Vh,err_trunc_st:err_trnc_end],label=L"$\hat{v}$",legend=:bottomleft,dpi=300)
    #plot!(sol.t[err_trunc_st:err_trnc_end],sol[NeurTrue.input,err_trunc_st:err_trnc_end])
    perrtrnc = plot(sol.t[err_trunc_st:err_trnc_end],
        abs.(sol[NeurTrue.V,err_trunc_st:err_trnc_end].-sol[Identifier.Vh,err_trunc_st:err_trnc_end]),label=L"|v-\hat{v}\,|",legend=:topleft,ylabel="[mV]")

    pinputtrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.input,err_trunc_st:err_trnc_end],label=L"$u$",ylabel=L"[$\mu$A]",legend=:bottomleft,dpi=300)
    
    pgCaLtrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gCaL,err_trunc_st:err_trnc_end]/3,label=L"$\mu_{\rm{CaL}}/N$",legend=:bottomleft,ylabel="[mS / cm^2]")
    meangCaLests = (sol[Identifier.θ[2*l+1],err_trunc_st:err_trnc_end].+sol[Identifier.θ[2*l+2],err_trunc_st:err_trnc_end].+sol[Identifier.θ[2*l+3],err_trunc_st:err_trnc_end])/3
    plot!(sol.t[err_trunc_st:err_trnc_end], meangCaLests,dpi=300,label=false)
    pgKCatrnc = plot(sol.t[err_trunc_st:err_trnc_end], sol[NeurTrue.gKCa,err_trunc_st:err_trnc_end]/3,label=L"$\mu_{\rm{KCa}}/N$",legend=:bottomleft,xlabel="t [ms]",ylabel="[mS / cm^2]")
    meangKCaests = (sol[Identifier.θ[4*l+1],err_trunc_st:err_trnc_end].+sol[Identifier.θ[4*l+2],err_trunc_st:err_trnc_end].+sol[Identifier.θ[4*l+3],err_trunc_st:err_trnc_end])/3
    plot!(sol.t[err_trunc_st:err_trnc_end], meangKCaests,dpi=300,label=false)
    
    fig2 = plot(perrtrnc,pgCaLtrnc,pgKCatrnc,layout=(3,1))
    savefig(fig2,"fig5_distredundant")

    earlier_start = Int((ramp_start-14000)/dt); earlier_end = Int((ramp_start-4000)/dt)
    pgCaTtrnc_spiking = plot(sol.t[earlier_start:earlier_end], sol[NeurTrue.gCaT,earlier_start:earlier_end]/l,label=L"$\mu_{\rm{CaT}}/N$",legend=:topleft,xlabel="t [ms]",ylabel="[mS / cm^2]")
    plot!(sol.t[earlier_start:earlier_end], sol[Identifier.θ[10],earlier_start:earlier_end],label=false)
    plot!(sol.t[earlier_start:earlier_end], sol[Identifier.θ[11],earlier_start:earlier_end],label=false)
    plot!(sol.t[earlier_start:earlier_end], sol[Identifier.θ[12],earlier_start:earlier_end],label=false)
    
    
    fig3 = plot(perrtrnc,pgCaTtrnc,layout=(2,1),dpi=300)
    savefig(fig3,"fig6_distredundant")
    
        savefig(pVtrnc,"pV_distredundant")
        savefig(perrtrunc,"perr_distredundant")
    
        pV=plot(sol, idxs=[NeurTrue.V])
    pVh=plot(sol,idxs=[Identifier.Vh])

    perr=plot(sol.t, sol[NeurTrue.V].-sol[Identifier.Vh])

    ptest=plot(sol, idxs=[Identifier.tmp[3]])

    function gCaT_ramp_l(t)
        if t < ramp_start
            return 3/l
        elseif ramp_start <= t < ramp_start+16000
            return (0.0001875*(t-ramp_start) + 3)/l
        else
            return 6/l
        end
    end

    p1=plot(sol, idxs=[Identifier.θ[1]], label=false)#,ylims=(0,0.01))
    plot!(sol, idxs=[Identifier.θ[2]])
    plot!(sol, idxs=[Identifier.θ[3]])
    hline!([100/l], width=1.5, label="gNa/$l")
    p2=plot(sol, idxs=[Identifier.θ[l+1]], label=false)
    hline!([65/l], width=1.5, label="gKd/$l")
    p3=plot(sol, idxs=[Identifier.θ[2*l+1]], label=false)
    hline!([25/l], width=1.5, label="gKCa/$l")
    p4=plot(sol, idxs=[Identifier.θ[3*l+1]], label=false)
    plot!(sol.t, gCaT_ramp_l, width=1.5, label="gCaT/$l")
    #p5=plot(sol, idxs=[Identifier.θ[5]], label=false)
    #hline!([0.5], width=1.5, label="gCaT")
    p5=plot(sol, idxs=[Identifier.θ[4*l+1]], label=false)
    hline!([0.3], width=1.5, label="gleak")

    j=length(sol.t); i=Int(floor(0.1*j));
    
    p1z=plot(sol.t[i:j], sol[Identifier.θ[1],i:j],xlabel="t [ms]",label=false);
    plot!(sol.t[i:j], sol[Identifier.θ[3],i:j],label=false);
    plot!(sol.t[i:j], sol[Identifier.θ[4],i:j],label=false);
    p2z=plot(sol.t[i:j], sol[Identifier.θ[l+1],i:j],xlabel="t [ms]",label=false)
    plot!(sol.t[i:j], sol[Identifier.θ[l+2],i:j],label=false)
    plot!(sol.t[i:j], sol[Identifier.θ[l+3],i:j],label=false)
    p3z=plot(sol.t[i:j], sol[Identifier.θ[2*l+1],i:j],xlabel="t [ms]",label=false)
    p4z=plot(sol.t[i:j], sol[Identifier.θ[3*l+1],i:j],xlabel="t [ms]",label=false);
    plot!(sol.t[i:j], sol[Identifier.θ[3*l+2],i:j],xlabel="t [ms]",label=false);
    plot!(sol.t[i:j], sol[Identifier.θ[3*l+3],i:j],xlabel="t [ms]",label=false);
    plot!(sol.t[i:j], gCaT_ramp_l, width=1.5, label="gCaT/$l");
    p5z=plot(sol.t[i:j], sol[Identifier.θ[4*l+1],i:j],xlabel="t [ms]",label=false)

    perrz = plot(sol.t[i:j], sol[NeurTrue.V,i:j] .- sol[Identifier.Vh,i:j]);

    pB = plot(p4,p5,layout=(2,1))
    pA = plot(p1,p2,p3,layout=(3,1))

    pAz = plot(p1z,p2z,p3z,layout=(3,1))
end