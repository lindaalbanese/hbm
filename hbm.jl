using PyPlot

function mctransition(β,ξ,σ,z) #prende uno stato del sistema in ingresso e tira fuori un nuovo stato
    σn=copy(σ)
    zn=copy(z)
    #aggiungere gibbs update per σ e z, compiti: calcolare P(σ|z,ξ,β) e P(z|σ,ξ,β)
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

