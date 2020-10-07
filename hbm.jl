using LinearAlgebra
using Statistics
using Random

function mctransition(β,ξ,σ;rng=Random.GLOBAL_RNG) #prende uno stato del sistema in ingresso e tira fuori un nuovo stato
    N, P = size(ξ)

    #ξ=(ξ.-mean(ξ))./std(ξ) #normalizzazione pesi

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
    ξ=(ξ.-mean(ξ))./std(ξ)
    β, chain=state
    N, P = size(ξ)
    mapped=map(chain) do (σ,z)
        mag=((sign.(ξ)')*σ)./length(σ) # calcola magnetizzazione sign. serve per evitare problemi se xi non corrisponde a +/- 1
        mag=abs.(mag) # rompo l'invarianza di gauge della magnetizzazione per poter mediare osservazioni indipendenti
        (M=mag, dlogZdξ= (β/sqrt(N))*σ*(z'), nM=sum(abs,mag))
    end
    K=keys(mapped[1])
    Dict([(k => mean(getindex.(mapped,k)[(skip+1):end])) for k in K])
end

function exp_data(β,ξ,σ) #calcolare l'aspettazione rispetto distrib dei dati 
    #ξ=(ξ.-mean(ξ))./std(ξ)
    N,P=size(ξ)
    fact=(ξ')*σ
    data=copy(ξ)
    for μ in 1:P, i in 1:N
        data[i, μ]=β/N*fact[μ]*σ[i]
    end
    #fact=reshape(fact, (P,1))
    #σ=reshape(σ, (N,1))
    #data=β/N*fact*σ'
    data
end

function exp_model(β,ξ,σ) #calcolo aspettazione rispetto distrib data dal modello
    N,P=size(ξ)
    ch=mcmc_chain(β, ξ, σ, 100)
    model=chain_estimators(ξ, ch, skip=10)
    model[:dlogZdξ],model[:M]
end

function CDstep(ϵ, β, ξ, σ)#passo di CD 
    N,P=size(ξ)
    data=exp_data(β, ξ, σ)  #media prob dati
    model,mag =exp_model(β, ξ, σ) #media prob modello
    ξ .+= ϵ.*(data.-model)
    #for  μ in 1:P, i in 1:N
    #    ξ[ i, μ] += ϵ*(data[i, μ]- model[i, μ]) #regola di aggiornamento
    #end
    mag
end


using ProgressMeter
#= 
function CDtest()
    ξ=sign.(randn(100,3)).*0.001

    σD=[sign.(randn(100)) for i in 1:2]
    @showprogress for j = 1:1000
        for i in eachindex(σD)
            CDstep(0.1,10.0,ξ,σD[i])
        end
    end
    σD,ξ
end



CDtest()
 =#
using PyPlot

pygui(true)

ϵ=0.01 #learnig rate
β= 1/1.1 #1/temp
r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
N=100
P=4
#creazione dataset
ξq=rand((-1.0,1.0),N,P) #P patterns

#ξq=rand(Binomial(1,0.5),(N,P))
#ξq=ifelse.(ξq .== 1, 1, -1)

p=0.10
data_rand = cat([sign.(rand(size(ξq)...) .- p) .* ξq for i=1:3]...,dims=3)

# ξ_iμ -> x_iμ / √(1+x_iμ^2)

function CDtest(data,ϵ,β)
    ϵ′=ϵ/β
    N, P, Nesempi = size(data)
    ξ=mean(data,dims=3)[:,:,1]
    normξ=norm(ξ)/sqrt(N*P)
    ξ.+=randn(size(ξ)...)*normξ*0.5

    refnorm=norm(data[:,:,1])
    matrices=Array{Float64,3}[]
    @showprogress for j = 1:500
        magμ=zeros(P,P,Nesempi)
        @inbounds for i in 1:Nesempi, μ = 1:P
            mag = CDstep(ϵ′,β,ξ,data[:,μ,i])
            for μ′ = 1:P
                magμ[μ,μ′,i]=mag[μ′]
            end
        end
        push!(matrices,magμ)
        μξ=mean(ξ)
        σξ=std(ξ)
        @. ξ = (ξ-μξ)/σξ
    end
    ξ,matrices
end

ξt,mat= CDtest(data_rand,0.1, 10.05)


imshow(mat[end][:,:,1]); colorbar()
