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

# translationally symmetric random product state
function randomize_initial_state(N, sites)
    
    mat = haar_random(2)       
    gates = ITensor[]
    for i=1:N
        s1 = sites[i]
        op = itensor(mat, s1', s1)
        push!(gates, op)
    end

    psi = MPS(sites, n -> isodd(n) ? "Up" : "Up")
    psi = ITensors.apply(gates, psi; cutoff = 1e-15)
    
    return psi
end


#Time evolution by translationally symmetric Hamiltonian
function TEBD(J::Float64, h::Float64, N::Int64, seed0::Int64)
    
    #
    Random.seed!(seed0)
    
    Jxx = 2*J*rand() - J
    Jyy = 2*J*rand() - J
    Jzz = 2*J*rand() - J
    Jxy = 2*J*rand() - J
    Jyx = 2*J*rand() - J
    Jyz = 2*J*rand() - J
    Jzy = 2*J*rand() - J
    Jzx = 2*J*rand() - J
    Jxz = 2*J*rand() - J
    
    hx = 2*h*rand() - h
    hy = 2*h*rand() - h
    hz = 2*h*rand() - h
    
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
        hj = ( Jxx * op("X", s1) * op("X", s2) 
             + Jyy * op("Y", s1) * op("Y", s2) 
             + Jzz * op("Z", s1) * op("Z", s2) 
             + Jxy * op("X", s1) * op("Y", s2) 
             + Jyx * op("Y", s1) * op("X", s2) 
             + Jyz * op("Y", s1) * op("Z", s2) 
             + Jzy * op("Z", s1) * op("Y", s2) 
             + Jzx * op("Z", s1) * op("X", s2) 
             + Jxz * op("X", s1) * op("Z", s2) 
             + (hx/2.) * op("X", s1) * op("I", s2) 
             + (hx/2.) * op("I", s1) * op("X", s2)
             + (hy/2.) * op("Y", s1) * op("I", s2) 
             + (hy/2.) * op("I", s1) * op("Y", s2)
             + (hz/2.) * op("Z", s1) * op("I", s2) 
             + (hz/2.) * op("I", s1) * op("Z", s2) )
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
