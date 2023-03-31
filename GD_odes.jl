using DifferentialEquations, Plots, Plots.PlotMeasures#, LaTeXStrings

# Return a value uniformly sampled from x +/- 100*err %
function x_sample(x, err)
    x*(1-err) + 2*err*x*rand()
end

## Model gating functions
# All activation and inactivation curves are defined by the Boltzman function
Xinf(V,A,B)=1/(1+exp((V+A)/B))

# All timeconstant curves are defined by the shifted Boltzman function
tauX(V,A,B,D,E)=A-B/(1+exp((V+D)/E))

# Sodium current
mNainf(V,r=25.,s=-5.) = Xinf(V,r,s); tau_mNa(V,r=100.) = tauX(V,0.75,0.5,r,-20.)
hNainf(V,r=40.,s=10.) = Xinf(V,r,s); tau_hNa(V,r=50.) = tauX(V,4.0,3.5,r,-20.)


# Potassium currents
mKdinf(V,r=15.) = Xinf(V,r,-10.); tau_mKd(V,r=30.) = tauX(V,5.0,4.5,r,-20.)

mAfinf(V,r=80) = Xinf(V,r,-10.); tau_mAf(V,r=100.) = tauX(V,0.75,0.5,r,-20.)
hAfinf(V,r=60) = Xinf(V,r,5.); tau_hAf(V,r=100.) = 10*tauX(V,0.75,0.5,r,-20.)

mAsinf(V,r=60) = Xinf(V,r,-10.); tau_mAs(V,r=100.) = 10*tauX(V,0.75,0.5,r,-20.)
hAsinf(V,r=20) = Xinf(V,r,5.); tau_hAs(V,r=100.) = 100*tauX(V,0.75,0.5,r,-20.)

mKCainf(Ca,r=-30.0) = Xinf(Ca,r,-10.); tau_mKCa = 500.


# Calcium currents
mCaLinf(V,r=45.,s=-5.) = Xinf(V,r,s); tau_mCaL(V,r=30.) = tauX(V,6.0,5.5,r,-20.)

mCaTinf(V,r=60,s=-5.) = Xinf(V,r,s); tau_mCaT(V,r=30.) = tauX(V,6.0,5.5,r,-20.)
hCaTinf(V,r=85,s=10.) = Xinf(V,r,s); tau_hCaT(V,r=30.) = 100*tauX(V,6.0,5.5,r,-20.)


# Cation current (H-current)
mHinf(V,r=85) = Xinf(V,r,10.); tau_mH(V,r=30.) = 50*tauX(V,6.0,5.5,r,-20.);

# Synapse
sinf(V,r=68,s=-2) = Xinf(V,r,s); stau(V,r=30.) = tauX(V,6.0,5.5,r,-20.)

tau_Ca = 500.

function init_neur(V0)
    mNa0=mNainf(V0) 
    hNa0=hNainf(V0) 
    mKd0=mKdinf(V0) 
    mAf0=mAfinf(V0) 
    hAf0=hAfinf(V0) 
    mAs0=mAsinf(V0) 
    hAs0=hAsinf(V0) 
    mCaL0=mCaLinf(V0) 
    mCaT0=mCaTinf(V0) 
    hCaT0=hCaTinf(V0) 
    mH0=mHinf(V0) 
    Ca0=(-αCa*gCaL*mCaL0*(V0-VCa))+(-β*gCaT*mCaT0*hCaT0*(V0-VCa))*2
    x0 = [V0 mNa0 hNa0 mKd0 mAf0 hAf0 mAs0 hAs0 mCaL0 mCaT0 hCaT0 mH0 Ca0]
end

## Simulation function in current-clamp mode
function CBM_ODE(du,u,p,t)
    # Stimulations parameters
    Iapp=p[1] # Amplitude of constant applied current
    I1=p[2] # Amplitude of first step input
    I2=p[3] # Amplitude of second step input
    ti1=p[4] # Starting time of first step input
    tf1=p[5] # Ending time of first step input
    ti2=p[6] # Starting time of second step input
    tf2=p[7] # Ending time of second step input
    
    # Maximal conductances
    gNa=p[8] # Sodium current maximal conductance
    gKd=p[9]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[10] # Fast A-type potassium current maximal conductance
    gAs=p[11] # Slow A-type potassium current maximal conductance
    gKCa=p[12] # Calcium-activated potassium current maximal conductance
    gCaL=p[13] # L-type calcium current maximal conductance
    gCaT=p[14] # T-type calcium current maximal conductance
    gH=p[15] # H-current maximal conductance
    gl=p[16] # Leak current maximal conductance

    half_acts=p[17]

    # Variables
    V=u[1] # Membrane potential
    mNa=u[2] # Sodium current activation
    hNa=u[3] # Sodium current inactivation
    mKd=u[4] # Delayed-rectifier potassium current activation
    mAf=u[5] # Fast A-type potassium current activation
    hAf=u[6] # Fast A-type potassium current inactivation
    mAs=u[7] # Slow A-type potassium current activation
    hAs=u[8] # Slow A-type potassium current inactivation
    mCaL=u[9] # L-type calcium current activation
    mCaT=u[10] # T-type calcium current activation
    hCaT=u[11] # T-type calcium current inactivation
    mH=u[12] # H current activation
    Ca=u[13] # Intracellular calcium concentration

    # ODEs
                    # Sodium current
    du[1] = (1/C) * (-gNa*mNa*hNa*(V-VNa) +
                    # Potassium Currents
                    -gKd*mKd*(V-VK) -gAf*mAf*hAf*(V-VK) -gAs*mAs*hAs*(V-VK) +
                    -gKCa*mKCainf(Ca,half_acts[12])*(V-VK) +
                    # Calcium currents
                    -gCaL*mCaL*(V-VCa) +
                    -gCaT*mCaT*hCaT*(V-VCa) +
                    # Cation current
                    -gH*mH*(V-VH) +
                    # Passive currents
                    -gl*(V-Vl) +
                    # Stimulation currents
                    +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))
    du[2] = (1/tau_mNa(V)) * (mNainf(V,half_acts[1]) - mNa)
    du[3] = (1/tau_hNa(V)) * (hNainf(V,half_acts[2]) - hNa)
    du[4] = (1/tau_mKd(V)) * (mKdinf(V,half_acts[3]) - mKd)
    du[5] = (1/tau_mAf(V)) * (mAfinf(V,half_acts[4]) - mAf)
    du[6] = (1/tau_hAf(V)) * (hAfinf(V,half_acts[5]) - hAf)
    du[7] = (1/tau_mAs(V)) * (mAsinf(V,half_acts[6]) - mAs)
    du[8] = (1/tau_hAs(V)) * (hAsinf(V,half_acts[7]) - hAs)
    du[9] = (1/tau_mCaL(V)) * (mCaLinf(V,half_acts[8]) - mCaL)
    du[10] = (1/tau_mCaT(V)) * (mCaTinf(V,half_acts[9]) - mCaT)
    du[11] = (1/tau_hCaT(V)) * (hCaTinf(V,half_acts[10]) - hCaT)
    du[12] = (1/tau_mH(V)) * (mHinf(V,half_acts[11]) - mH)
    du[13] = (1/tau_Ca) * ((-αCa*gCaL*mCaL*(V-VCa))+(-β*gCaT*mCaT*hCaT*(V-VCa)) - Ca) 
end


## Stimulation function
heaviside(t)=(1+sign(t))/2 # Unit step function
pulse(t,ti,tf)=heaviside(t-ti)-heaviside(t-tf) # Pulse function

function CBM_observer!(du,u,p,t)
    # Stimulations parameters
    Iapp=p[1] # Amplitude of constant applied current
    I1=p[2] # Amplitude of first step input
    I2=p[3] # Amplitude of second step input
    ti1=p[4] # Starting time of first step input
    tf1=p[5] # Ending time of first step input
    ti2=p[6] # Starting time of second step input
    tf2=p[7] # Ending time of second step input
    
    # Maximal conductances
    gNa=p[8] # Sodium current maximal conductance
    gKd=p[9]  # Delayed-rectifier potassium current maximal conductance
    gAf=p[10] # Fast A-type potassium current maximal conductance
    gAs=p[11] # Slow A-type potassium current maximal conductance
    gKCa=p[12] # Calcium-activated potassium current maximal conductance
    gCaL=p[13] # L-type calcium current maximal conductance
    gCaT=p[14] # T-type calcium current maximal conductance
    gH=p[15] # H-current maximal conductance
    gl=p[16] # Leak current maximal conductance
    
    halfs_acts = p[17]

    # Variables
    V=u[1] # Membrane potential
    mNa=u[2] # Sodium current activation
    hNa=u[3] # Sodium current inactivation
    mKd=u[4] # Delayed-rectifier potassium current activation
    mAf=u[5] # Fast A-type potassium current activation
    hAf=u[6] # Fast A-type potassium current inactivation
    mAs=u[7] # Slow A-type potassium current activation
    hAs=u[8] # Slow A-type potassium current inactivation
    mCaL=u[9] # L-type calcium current activation
    mCaT=u[10] # T-type calcium current activation
    hCaT=u[11] # T-type calcium current inactivation
    mH=u[12] # H current activation
    Ca=u[13] # Intracellular calcium concentration
    
    θ = [gCaL gCaT]
    ϕ = 1/C*[-mCaL*(V-VCa) ...
            -mCaT*hCaT*(V-VCa)];
        
                # Sodium current
    b = (1/C) * (-gNa*mNa*hNa*(V-VNa) +
                # Potassium Currents
                -gKd*mKd*(V-VK) -gAf*mAf*hAf*(V-VK) -gAs*mAs*hAs*(V-VK) +
                -gKCa*mKCainf(Ca)*(V-VK) +
                # Cation current
                -gH*mH*(V-VH) +
                # Passive currents
                -gl*(V-Vl) +
                # Stimulation currents
                +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))
    du[1] = dot(ϕ,θ) + b

    # Internal dynamics
    du[2] = (1/tau_mNa(V)) * (mNainf(V) - mNa)
    du[3] = (1/tau_hNa(V)) * (hNainf(V) - hNa)
    du[4] = (1/tau_mKd(V)) * (mKdinf(V) - mKd)
    du[5] = (1/tau_mAf(V)) * (mAfinf(V) - mAf)
    du[6] = (1/tau_hAf(V)) * (hAfinf(V) - hAf)
    du[7] = (1/tau_mAs(V)) * (mAsinf(V) - mAs)
    du[8] = (1/tau_hAs(V)) * (hAsinf(V) - hAs)
    du[9] = (1/tau_mCaL(V)) * (mCaLinf(V) - mCaL)
    du[10] = (1/tau_mCaT(V)) * (mCaTinf(V) - mCaT)
    du[11] = (1/tau_hCaT(V)) * (hCaTinf(V) - hCaT)
    du[12] = (1/tau_mH(V)) * (mHinf(V) - mH)
    du[13] = (1/tau_Ca) * ((-αCa*gCaL*mCaL*(V-VCa))+(-β*gCaT*mCaT*hCaT*(V-VCa)) - Ca) 

    # Adaptive observer
    Vh=u[14] # Membrane potential
    mNah=u[15] # Sodium current activation
    hNah=u[16] # Sodium current inactivation
    mKdh=u[17] # Delayed-rectifier potassium current activation
    mAfh=u[18] # Fast A-type potassium current activation
    hAfh=u[19] # Fast A-type potassium current inactivation
    mAsh=u[20] # Slow A-type potassium current activation
    hAsh=u[21] # Slow A-type potassium current inactivation
    mCaLh=u[22] # L-type calcium current activation
    mCaTh=u[23] # T-type calcium current activation
    hCaTh=u[24] # T-type calcium current inactivation
    mHh=u[25] # H current activation
    Cah=u[26] # Intracellular calcium concentration
    θ̂= u[27:28]
    P = reshape(u[28+1:28+4],2,2);    
    P = (P+P')/2
    Ψ = u[28+4+1:28+4+2]

    ϕ̂ = 1/C*[-mCaLh * (V-VCa) ...
            -mCaTh * hCaTh * (V-VCa)];

    bh = (1/C) * (-gNa*mNah*hNah*(V-VNa) +
    # Potassium Currents
    -gKd*mKdh*(V-VK) -gAf*mAfh*hAfh*(V-VK) -gAs*mAsh*hAsh*(V-VK) +
    -gKCa*mKCainf(Cah,half_acts[12])*(V-VK) +
    # Cation current
    -gH*mHh*(V-VH) +
    # Passive currents
    -gl*(V-Vl) +
    # Stimulation currents
    +Iapp(t) + I1*pulse(t,ti1,tf1) + I2*pulse(t,ti2,tf2))

    # dV^
    du[14] = dot(ϕ̂,θ̂) + bh + γ*(1+Ψ'*P*Ψ)*(V-Vh)

    # Internal dynamics
    du[15] = (1/tau_mNa(V)) * (mNainf(V,half_acts[1]) - mNah)
    du[16] = (1/tau_hNa(V)) * (hNainf(V,half_acts[2]) - hNah)
    du[17] = (1/tau_mKd(V)) * (mKdinf(V,half_acts[3]) - mKdh)
    du[18] = (1/tau_mAf(V)) * (mAfinf(V,half_acts[4]) - mAfh)
    du[19] = (1/tau_hAf(V)) * (hAfinf(V,half_acts[5]) - hAfh)
    du[20] = (1/tau_mAs(V)) * (mAsinf(V,half_acts[6]) - mAsh)
    du[21] = (1/tau_hAs(V)) * (hAsinf(V,half_acts[7]) - hAsh)
    du[22] = (1/tau_mCaL(V)) * (mCaLinf(V,half_acts[8]) - mCaLh)
    du[23] = (1/tau_mCaT(V)) * (mCaTinf(V,half_acts[9]) - mCaTh)
    du[24] = (1/tau_hCaT(V)) * (hCaTinf(V,half_acts[10]) - hCaTh)
    du[25] = (1/tau_mH(V)) * (mHinf(V,half_acts[11]) - mHh)
    du[26] = (1/tau_Ca) * ((-αCa*gCaL*mCaLh*(V-VCa))+(-β*gCaT*mCaTh*hCaTh*(V-VCa)) - Cah) 

    du[27:28]= γ*P*Ψ*(V-Vh); # dθ̂ 
    du[28+4+1:28+4+2] = -γ*Ψ + ϕ̂;  # dΨ
    dP = α*P - ((P*Ψ)*(P*Ψ)');
    dP = (dP+dP')/2;
    du[28+1:28+4] = dP[:]
end

