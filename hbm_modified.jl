#MODIFICA RISPETTO HBM:
# ξ_iμ -> x_iμ / √(1+x_iμ^2)


using LinearAlgebra
using Statistics
using Random

function mctransition(β,x,σ;rng=Random.GLOBAL_RNG) #prende uno stato del sistema in ingresso e tira fuori un nuovo stato
    N, P = size(x)
    ξ= x./sqrt.((ones(size(x)).+(x).^2))
    #ξ=(ξ.-mean(ξ))./std(ξ) #normalizzazione pesi
    length(σ) == N || error("wrong dimension for σ")
    z = ((ξ')*σ)./sqrt(N) .+ randn(P)./sqrt(β) # calcoliamo le z usando la P(z|σ,ξ,β).
    fact = ξ*z # si può fare più semplicemente cosi
    prob_s = @. 1 / (1 + exp(-2*β*fact/sqrt(N))) # calcolo le probabilità che gli spin risultino +1 usando la P(σ_i|z,ξ,β).
    σn = sign.(prob_s .- rand(N)) # campiono i nuovi spin con le probabilità prob_s
    zn = ((ξ')*σn)./sqrt(N) .+ randn(P)./sqrt(β) # calcolo gli z appartenenti alla nuova configurazione sfruttando la P(z|σn,ξ,β).
    (σn, zn)
end

function mcmc_chain(β,x,σ,N;rng=Random.GLOBAL_RNG) #prende uno stato in ingresso e tira fuori una catena di stati
    chain=fill(mctransition(β,x,σ;rng),N)
    for i = 2:N
        chain[i] = mctransition(β,x,chain[i-1][1];rng)
    end
    (β=β,chain=chain)
end

function chain_estimators(x,state;skip=0)
    #ξ=(ξ.-mean(ξ))./std(ξ)
    β, chain=state
    N, P = size(x)
    ξ= x#./sqrt.((ones(size(x)).+(x).^2))
    mapped=map(chain) do (σ,z)
        mag=((sign.(ξ)')*σ)./length(σ) # calcola magnetizzazione sign. serve per evitare problemi se xi non corrisponde a +/- 1
        mag=abs.(mag) # rompo l'invarianza di gauge della magnetizzazione per poter mediare osservazioni indipendenti
        (M=mag, dlogZdξ= (β/sqrt(N))*σ*(z'), nM=sum(abs,mag))
    end
    K=keys(mapped[1])
    Dict([(k => mean(getindex.(mapped,k)[(skip+1):end])) for k in K])
end

function exp_data(β,x,σ) #calcolare l'aspettazione rispetto distrib dei dati 
    #ξ=(ξ.-mean(ξ))./std(ξ)
    N,P=size(x)
    ξ= x./sqrt.((ones(size(x)).+(x).^2))
    fact=(ξ')*σ
    data=copy(ξ)
    for μ in 1:P, i in 1:N
        data[i, μ]=β/N*fact[μ]*σ[i]*1/sqrt((1+x[i,μ]^2)^3)
    end
    #fact=reshape(fact, (P,1))
    #σ=reshape(σ, (N,1))
    #data=β/N*fact*σ'
    data
end

function exp_model(β,x,σ) #calcolo aspettazione rispetto distrib data dal modello
    N,P=size(x)
    ch=mcmc_chain(β, x, σ, 100)
    model=chain_estimators(x, ch, skip=10)
    model[:dlogZdξ],model[:M]
end

function CDstep(ϵ, β, x, σ)#passo di CD 
    N,P=size(x)
    data=exp_data(β, x, σ)  #media prob dati
    model,mag =exp_model(β, x, σ) #media prob modello
    x .+= ϵ.*(data.-model)
    mag 
end


using ProgressMeter

#= function CDtest()
    x=sign.(randn(100,3)).*0.001

    σD=[sign.(randn(100)) for i in 1:2]
    @showprogress for j = 1:1000
        for i in eachindex(σD)
            CDstep(0.01,10.0,x,σD[i])
        end
    end
    ξt= x./sqrt.((ones(size(x)).+(x).^2))
    σD,ξt
end


CDtest() =#


 function overlap_weigths(ξ, σ) #calcolo dell'overlap tra i pesi
    N,P=size(ξ)
    over=zeros((P,P))
    for μ in 1:P
        for ν in 1:P
            p1=norm(ξ[:,μ],2)
            p2=norm(σ[:,ν],2)
            over[μ,ν]= 1/(p1*p2)*sum(ξ[i,μ]*σ[i,ν] for i in 1:N)
        end
    end
    over
end


using PyPlot

pygui(true)


function CDtest(data,ϵ,β)
    ϵ′=ϵ#/β
    N, P, Nesempi = size(data)
    #x=mean(data,dims=3)[:,:,1]
    x=(randn(N,P))#.*0.001
    #normξ=norm(x)/sqrt(N*P)
    #x.+=randn(size(x)...)*normξ*0.5
    #refnorm=norm(data[:,:,1])
    matrices=Array{Float64,3}[]
    @showprogress for j = 1:300
        magμ=zeros(P,P,Nesempi)
        @inbounds for i in 1:Nesempi, μ = 1:P
            mag = CDstep(ϵ′,β,x,data[:,μ,i])
            for μ′ = 1:P
                magμ[μ,μ′,i]=mag[μ′]
            end
        end
        push!(matrices,magμ)
        #μξ=mean(ξ)
        #σξ=std(ξ)
        #@. ξ = (ξ-μξ)/σξ
    end
    #ξ= x./sqrt.((ones(size(x)).+(x).^2))
    ξ,matrices
end

function statistics(mat, thresh, data)
    good=0
    wrong=0
    spur=0
    #= good_data=Array{Int,1}[]
    wrong_data=Array{Int,1}[]
    spur_data=Array{Int,1}[] =#
    for i in 1:size(mat)[3]
        #matt=mat[:,:,i]
        for j in 1:P
            if findmax(mat[j,:,i])[1] > thresh
                if findmax(mat[j,:,i])[2] == j 
                    good+=1
                    #push!(good_data, data[:,j,i])
                else
                    wrong+=1
                    #push!(wrong_data, data[:,j,i])
                end
            else
                spur+=1
                #push!(spur_data, data[:,j,i])
            end
        end
    end
    good, wrong, spur
end

ϵ=0.01 #learnig rate
β= 1/0.2 #1/temp
N=100
P=4
#creazione dataset
ξq=rand((-1.0,1.0),N,P) #P patterns
p=0.10
data_rand = cat([sign.(rand(size(ξq)...) .- p) .* ξq for i=1:10]...,dims=3)

ξ=mean(data_rand,dims=3)[:,:,1] #media dataset

xt,mat= CDtest(data_rand,ϵ, β)
ξt= xt./sqrt.((ones(size(xt)).+(xt).^2)) #pesi finali

#overlap tra pesi appresi e pattern medi
pygui(true)
figure()
imshow(overlap_weigths(ξ, ξt)); colorbar()
title("overlap-learning")

#= pygui(true)
imshow(ξt'*ξ./N); colorbar() =#

#valutare retrieval -> un solo passo di CD 
N, P, Nesempi = size(data_rand)
mat2=Array{Float64,3}[]
magμ=zeros(P,P,Nesempi)
@inbounds for i in 1:Nesempi, μ = 1:P
    mag = CDstep(ϵ,β,ξt,data_rand[:,μ,i])
    for μ′ = 1:P
        magμ[μ,μ′,i]=mag[μ′]
    end
end
push!(mat2,magμ)

pygui(true)
figure()
imshow(mat2[1][:,:,2]); colorbar()
title("magnetization")

matt=mat2[1]
good1, wrong1, spur1=statistics(matt, 0.65, data_rand)
