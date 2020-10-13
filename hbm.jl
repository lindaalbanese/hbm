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
    #ξ=(ξ.-mean(ξ))./std(ξ)
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
    mag #SE USI IN HBM
    #ξ #SE USI IN CHECK-DATA
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
    ξ=mean(data,dims=3)[:,:,1]
    #normξ=norm(ξ)/sqrt(N*P)
    #ξ.+=randn(size(ξ)...)*normξ*0.5
    #refnorm=norm(data[:,:,1])
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
#r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
N=100
P=4
#creazione dataset
ξq=rand((-1.0,1.0),N,P) #P patterns

#ξq=rand(Binomial(1,0.5),(N,P))
#ξq=ifelse.(ξq .== 1, 1, -1)

p=0.10
data_rand = cat([sign.(rand(size(ξq)...) .- p) .* ξq for i=1:50]...,dims=3)

# ξ_iμ -> x_iμ / √(1+x_iμ^2)

ξ=mean(data_rand,dims=3)[:,:,1]

ξt,mat= CDtest(data_rand,ϵ, β)

pygui(true)
figure()
imshow(mat[end][:,:,1]); colorbar()
title("magnetization")


pygui(true)
figure()
imshow(overlap_weigths(ξ, ξt)); colorbar()
title("overlap-learning")

#= pygui(true)
imshow(ξt'*ξ./N); colorbar() =#

matt=mat[end]
good1, wrong1, spur1=statistics(matt, 0.65, data_rand)

############################
#LOOP
#= ϵ=0.01 #learnig rate
values=[(0.01, 1/0.2), (0.02, 1/0.2), (0.03, 1/0.2), (0.04, 1/0.2), (0.05, 1/0.2), (0.06, 1/0.2), (0.07, 1/0.2), (0.08, 1/0.2), (0.09, 1/0.2), (0.10, 1/0.2), (0.11, 1/0.2) ]
P=4
good_data=[]
difference=[]
for (α, β) in values 
    N=Int(floor(P/α))

    ξq=rand((-1.0,1.0),N,P) #P patterns
    p=0.10
    data_rand = cat([sign.(rand(size(ξq)...) .- p) .* ξq for i=1:10]...,dims=3)
    ξ=mean(data_rand,dims=3)[:,:,1]

    ξt,mat= CDtest(data_rand,ϵ, β)
    matt=mat[end]
    good1, wrong1, spur1=statistics(matt, 0.65, data_rand)
    push!(good_data, good1)
    push!(difference,mean(diag(overlap_weigths(ξ, ξt)))-abs(1/(P*P-P)*sum(overlap_weigths(ξ, ξt)-Diagonal(diag(overlap_weigths(ξ, ξt))))))
end

pygui(true)
figure()
plot([0.01,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], good_data, "o-")
title("good")

pygui(true)
figure()
plot([0.01,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], difference, "o-")
title("difference")
 =#

############################
#LOOP CON BETA  
#= ϵ=0.01 #learnig rate
values=[(0.04, 1/0.2), (0.04, 1/0.4), (0.04, 1/0.8), (0.04, 1/1.2), (0.04, 1/1.5), (0.04, 1/1.8), (0.04, 1/2), (0.04, 1/5), (0.04, 1/10)]
P=4
N=100
ξq=rand((-1.0,1.0),N,P) #P patterns
p=0.10
data_rand = cat([sign.(rand(size(ξq)...) .- p) .* ξq for i=1:10]...,dims=3)
ξ=mean(data_rand,dims=3)[:,:,1]

good_data=[]
difference=[]
for (α, β) in values 
    ξt,mat= CDtest(data_rand,ϵ, β)
    matt=mat[end]
    good1, wrong1, spur1=statistics(matt, 0.65, data_rand)
    push!(good_data, good1)
    #push!(difference,sqrt(mean((diag(overlap_weigths(ξ, ξt))).^2))-sqrt(1/(P*P-P)*sum((overlap_weigths(ξ, ξt)-Diagonal(diag(overlap_weigths(ξ, ξt)).^2)))))
    #push!(difference,mean(diag(overlap_weigths(ξ, ξt)))-abs(1/(P*P-P)*sum(overlap_weigths(ξ, ξt)-Diagonal(diag(overlap_weigths(ξ, ξt))))))
    push!(difference,mean(diag(overlap_weigths(ξ, ξt)).^2)- (mean((overlap_weigths(ξ, ξt)-Diagonal(diag(overlap_weigths(ξ, ξt)))).^2))*(P*P)/(P*P-P))
end

#= beta=[]
for i in 1: size(values)[1]
    append!(beta, values[i][2])
end =#

pygui(true)
figure()
plot([0.2,0.4, 0.8, 1.2, 1.5, 1.8, 2, 5, 10], good_data, "o-")
title("good")

pygui(true)
figure()
plot([0.2,0.4, 0.8, 1.2, 1.5, 1.8, 2, 5, 10], difference, "o-")
title("difference") =#

