# Haar random
# ref：https://pennylane.ai/qml/demos/tutorial_haar_measure
function haar_random(dim::Int)    
                            
    #Random.seed!(123456789+seed)
                            
    a=randn(dim,dim)
    b=randn(dim,dim)
    c = a+b*im
                            
    # QR decomposition
    QRFactorization = qr(c)

    # Q matrix
    Q = QRFactorization.Q

    # R matrix
    R = QRFactorization.R

    # Lambda matrix
    Lambda = zeros(ComplexF64, dim, dim)
    for i=1:dim
        Lambda[i,i] = R[i,i]/abs(R[i,i])
    end

    # haar random
    U = Q*Lambda
    return U            
end


#non-translationally symmetric random product state
function randomize_initial_state(N, sites)
         
    gates = ITensor[]
    for i=1:N
        mat = haar_random(2) 
        s1 = sites[i]
        op = itensor(mat, s1', s1)
        push!(gates, op)
    end

    psi = MPS(sites, n -> isodd(n) ? "Up" : "Up")
    psi = ITensors.apply(gates, psi; cutoff = 1e-15)
    
    return psi
end


#Time evolution
function TEBD(J::Float64, h::Float64, N::Int64, seed0::Int64)
    
    #
    Random.seed!(seed0)
    
    Jxx = 2*J*rand(N) - J*ones(N)
    Jyy = 2*J*rand(N) - J*ones(N)
    Jzz = 2*J*rand(N) - J*ones(N)
    Jxy = 2*J*rand(N) - J*ones(N)
    Jyx = 2*J*rand(N) - J*ones(N)
    Jyz = 2*J*rand(N) - J*ones(N)
    Jzy = 2*J*rand(N) - J*ones(N)
    Jzx = 2*J*rand(N) - J*ones(N)
    Jxz = 2*J*rand(N) - J*ones(N)
    
    hx = 2*h*rand(N) - h*ones(N)
    hy = 2*h*rand(N) - h*ones(N)
    hz = 2*h*rand(N) - h*ones(N)
    
    #
    tau0 = 0.01
    ttotal = 0.5
    step = ceil(ttotal/tau0)
    tau = ttotal/step 
    
    #
    sites = siteinds("S=1/2", N)
    c = div(N, 2) # center site
    cutoff = 1E-10

    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j in 1:N
        s1 = sites[j]
        s2 = sites[j%N + 1]
        hj = ( Jxx[j] * op("X", s1) * op("X", s2) 
             + Jyy[j] * op("Y", s1) * op("Y", s2) 
             + Jzz[j] * op("Z", s1) * op("Z", s2) 
             + Jxy[j] * op("X", s1) * op("Y", s2) 
             + Jyx[j] * op("Y", s1) * op("X", s2) 
             + Jyz[j] * op("Y", s1) * op("Z", s2) 
             + Jzy[j] * op("Z", s1) * op("Y", s2) 
             + Jzx[j] * op("Z", s1) * op("X", s2) 
             + Jxz[j] * op("X", s1) * op("Z", s2) 
             + (hx[j]/2.) * op("X", s1) * op("I", s2) 
             + (hx[j%N+1]/2.) * op("I", s1) * op("X", s2)
             + (hy[j]/2.) * op("Y", s1) * op("I", s2) 
             + (hy[j%N+1]/2.) * op("I", s1) * op("Y", s2)
             + (hz[j]/2.) * op("Z", s1) * op("I", s2) 
             + (hz[j%N+1]/2.) * op("I", s1) * op("Z", s2) )
        #https://itensor.github.io/ITensors.jl/dev/IncludedSiteTypes.html#%22Qubit%22-SiteType
        Gj = exp(-im * tau * hj / 2.0)
        push!(gates, Gj)
    end
    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    append!(gates, reverse(gates))

    
    # Initialize psi to be a random product state
    #psi = MPS(sites, n -> isodd(n) ? "Up" : "Up")
    psi = randomize_initial_state(N, sites)

    # Compute and print <Sz> at each time step
    # then apply the gates to go to the next time
    for t in 0.0:tau:ttotal        
        t≈ttotal && break
        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end
    
    return psi
end


# Observables
function XY(psi::MPS, a, b)
    
    sites = siteinds(psi) 
    
    os = OpSum()
    os += "X",a,"Y",b
    H = MPO(os,sites)
    xy = inner(psi',H,psi)
    
    return real(xy)
end

function XXYY_nonlinear(psi::MPS, a, b, c, d)
    
    sites = siteinds(psi) 
    
    os = OpSum()
    os += "X",a,"X",b
    H = MPO(os,sites)
    xx = inner(psi',H,psi)
    
    os = OpSum()
    os += "Y",c,"Y",d
    H = MPO(os,sites)
    yy = inner(psi',H,psi)
    
    return real(xx*yy)
end

function expXY(psi::MPS, a, b)
    
    sites = siteinds(psi) 
    
    os = OpSum()
    os += "X",a,"Y",b
    H = MPO(os,sites)
    xy = inner(psi',H,psi)
    z = exp(xy)
    
    return real(z)
end

function EE(psi::MPS, a::Int, b::Int)

    @assert b == a + 1

    psi = copy(psi)
    ITensors.orthogonalize!(psi, a)

    sites = siteinds(psi)

    sa = sites[a]
    sb = sites[b]

    AA = psi[a] * psi[b]

    rho = prime(AA, sa, sb) * dag(AA)

    Cbra = combiner(prime(sa), prime(sb))
    Cket = combiner(sa, sb)

    M = Cbra * rho * Cket

    #println("inds(M) = ", inds(M))

    rho_mat = matrix(M)

    #println("size(rho_mat) = ", size(rho_mat))

    vals = eigvals(Hermitian((rho_mat + rho_mat')/2))

    vals .= max.(vals, 0.0)

    #println("trace = ", sum(vals))

    vals ./= sum(vals)

    return -sum(p * log(p) for p in vals if p > 1e-15)
end

function EE_pauli(psi::MPS, a::Int, b::Int)

    sites = siteinds(psi)

    # Pauli matrices（2x2）
    II = ComplexF64[1 0; 0 1]
    X  = ComplexF64[0 1; 1 0]
    Y  = ComplexF64[0 -im; im 0]
    Z  = ComplexF64[1 0; 0 -1]

    # expectation values of Pauli operators
    os = OpSum(); os += "X", b;      H = MPO(os, sites); IX = inner(psi', H, psi)
    os = OpSum(); os += "Y", b;      H = MPO(os, sites); IY = inner(psi', H, psi)
    os = OpSum(); os += "Z", b;      H = MPO(os, sites); IZ = inner(psi', H, psi)

    os = OpSum(); os += "X", a;      H = MPO(os, sites); XI = inner(psi', H, psi)
    os = OpSum(); os += "X",a,"X",b; H = MPO(os, sites); XX = inner(psi', H, psi)
    os = OpSum(); os += "X",a,"Y",b; H = MPO(os, sites); XY = inner(psi', H, psi)
    os = OpSum(); os += "X",a,"Z",b; H = MPO(os, sites); XZ = inner(psi', H, psi)

    os = OpSum(); os += "Y", a;      H = MPO(os, sites); YI = inner(psi', H, psi)
    os = OpSum(); os += "Y",a,"X",b; H = MPO(os, sites); YX = inner(psi', H, psi)
    os = OpSum(); os += "Y",a,"Y",b; H = MPO(os, sites); YY = inner(psi', H, psi)
    os = OpSum(); os += "Y",a,"Z",b; H = MPO(os, sites); YZ = inner(psi', H, psi)

    os = OpSum(); os += "Z", a;      H = MPO(os, sites); ZI = inner(psi', H, psi)
    os = OpSum(); os += "Z",a,"X",b; H = MPO(os, sites); ZX = inner(psi', H, psi)
    os = OpSum(); os += "Z",a,"Y",b; H = MPO(os, sites); ZY = inner(psi', H, psi)
    os = OpSum(); os += "Z",a,"Z",b; H = MPO(os, sites); ZZ = inner(psi', H, psi)

    II_val = inner(psi, psi)  

    # ρ = (1/4) Σ_{μν} <σ^μ_a σ^ν_b> σ^μ ⊗ σ^ν
    rho_mat = (1/4) * (
        II_val * kron(II, II) +
        IX     * kron(II, X)  +
        IY     * kron(II, Y)  +
        IZ     * kron(II, Z)  +
        XI     * kron(X,  II) +
        XX     * kron(X,  X)  +
        XY     * kron(X,  Y)  +
        XZ     * kron(X,  Z)  +
        YI     * kron(Y,  II) +
        YX     * kron(Y,  X)  +
        YY     * kron(Y,  Y)  +
        YZ     * kron(Y,  Z)  +
        ZI     * kron(Z,  II) +
        ZX     * kron(Z,  X)  +
        ZY     * kron(Z,  Y)  +
        ZZ     * kron(Z,  Z)
    )

    # Hermitianization and eigenvalue calculation
    rho_mat = 0.5 * (rho_mat + rho_mat')
    vals = eigvals(Hermitian(rho_mat))
    vals .= max.(vals, 0.0)
    vals ./= sum(vals)

    return -sum(p * log(p) for p in vals if p > 1e-15)
end


# Function to project ρ̂ onto a physical state (minimizing Frobenius norm)
function project_to_physical(rho_est::Matrix{ComplexF64})
    n = size(rho_est, 1)
    
    σ = HermitianSemidefinite(n)
    diff = σ - rho_est
    obj = sumsquares(real(diff)) + sumsquares(imag(diff))
    
    prob = minimize(obj, [tr(σ) == 1])
    
    solve!(prob, SCS.Optimizer)
    
    return Hermitian(evaluate(σ))
end

# Entanglement entropy
function entanglement_entropy(rho)
    vals = eigvals(Hermitian(rho))
    vals = max.(real.(vals), 0.0)  # Clipping
    vals ./= sum(vals)              # Renormalization
    return -sum(v -> v > 1e-12 ? v * log(v) : 0.0, vals)
end