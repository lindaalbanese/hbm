using PyPlot

function mctransition(β,ξ,σ,z) #prende uno stato del sistema in ingresso e tira fuori un nuovo stato
    σn=copy(σ)
    zn=copy(z)
    (P,N)=lenght(zn)
    ξn=copy(z)

    fact=empty(N)
    for i = 1:N
        fact[i]=sum(ξn[mu][i]*zn[mu] for mu=1:P)
    prob_si=empty(N)
    for i = 1:N
        prob_si[i]= exp(β*σn[i]*fact[i])/(exp(β*fact[i]) + exp(-β*fact[i]))
    prob_s = prod(prob_si) #P(σn|zn,ξn,β) 
    avg=empty(P)
    for mu=1:P
        avg[mu]= sum(ξn[mu][i]*σn[i] for i =1:N) 
    prob_z = (β/(2*π))^(P/2)*exp(-β/2*sum((zn[mu] - avg[mu])^2) for mu =1:P) #P(zn|σn,ξn,β)
    #aggiungere gibbs update per σ e z, compiti: calcolare P(σ|z,ξ,β) e P(z|σ,ξ,β)
    
    Ham = 1/2*sum(zn^2[mu]for mu=1:P) - sum(σn[i]*fact[i] for i=1:N) #H(σ, z, ξ) 
    (σn,zn)
end

function mcmc_chain(β,ξ,σ,z,N) #prende uno stato in ingresso e tira fuori una catena di stati
    chain=fill((σ,z),N)
    chain[1] = mctransition(β,ξ,σ,z)
    for i = 2:N
        chain[i] = mctransition(β,ξ,chain[i-1]...)
    end
    chain
end

