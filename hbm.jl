using LinearAlgebra
using Statistics
using Random
function mctransition(β,ξ,σ;rng=Random.GLOBAL_RNG) #prende uno stato del sistema in ingresso e tira fuori un nuovo stato
    N, P = size(ξ)
    length(σ) == N || error("wrong dimension for σ")

    z = ((ξ')*σ)./sqrt(N) .+ randn(P)./sqrt(β) # calcoliamo le z usando la P(z|σ,ξ,β).
    fact = ξ*z # si può fare più semplicemente cosi
    prob_s = @. 1 / (1 + exp(-2*β*fact/sqrt(N))) # calcolo le probabilità che gli spin risultino +1 usando la P(σ_i|z,ξ,β).
    σn = sign.(prob_s .- rand(N)) # campiono i nuovi spin con le probabilità prob_s
    zn = ((ξ')*σn)./sqrt(N) .+ randn(P)./sqrt(β) # calcolo gli z appartenenti alla nuova configurazione sfruttando la P(z|σn,ξ,β).
    (σn, zn)
end

function mcmc_chain(β,ξ,σ,N;rng=Random.GLOBAL_RNG) #prende uno stato in ingresso e tira fuori una catena di stati
    chain=fill(mctransition(β,ξ,σ;rng),N)
    for i = 2:N
        chain[i] = mctransition(β,ξ,chain[i-1][1];rng)
    end
    (β=β,chain=chain)
end

function chain_estimators(ξ,state;skip=0)
    β, chain=state
    N, P = size(ξ)
    mapped=map(chain) do (σ,z)
        mag=((sign.(ξ)')*σ)./length(σ) # calcola magnetizzazione sign. serve per evitare problemi se xi non corrisponde a +/- 1
        mag=@.(mag*sign(σ[1]*ξ[1,:])) # rompo l'invarianza di gauge della magnetizzazione per poter mediare osservazioni indipendenti
        (M=mag, dlogZdξ= (β/sqrt(N))*σ*(z'), nM=sum(abs,mag))
    end
    K=keys(mapped[1])
    Dict([(k => mean(getindex.(mapped,k)[(skip+1):end])) for k in K])
end
