cd(@__DIR__)
include("hbm.jl")

using PyPlot 
#using Statistics

using LinearAlgebra
using Statistics

using ProgressMeter


function dataset(ξ, q, tot_ex)
    N,P=size(ξ)
    data=zeros(tot_ex, P, N)
    for ex in 1: tot_ex, μ in 1:P
        data[ex,μ,:]=ξ[:,μ,:] #esempio con un dato rumore q 
        flip=floor(Int, q*N) #numero di spin da flippare
        for i in 1:flip
            s=rand((1:N)) #scelgo random quali spin flippare
            data[ex,μ,s]=-data[ex,μ,s]
        end
    end
    data
end 

function magn_mean(ϵ,r, β, data, ξ) #calcolo media delle magnetizzazione trovate nel test
    runs=100
    V=@showprogress [test(ϵ,r, β, data, ξ) for i in 1:runs]
    (reps=r,
    m_v=mean(getfield.(v,:m_v) for v in V)) #getfield da v prende solo i valori contrassegnati m_v
end


function magnet(ξ, β, σ)
    N,P =size(ξ)
    mean(begin
    (σ, z)=mctransition(β,ξ,σ)
    mag=((sign.(ξ)')*σ)./length(σ) # calcola magnetizzazione sign. serve per evitare problemi se xi non corrisponde a +/- 1
    mag=@.(mag*sign(σ[1]*ξ[1,:])) # rompo l'invarianza di gauge della magnetizzazione per poter mediare osservazioni indipendenti
    end for i in 1:30)
end

function test(ϵ,reps, β, data, ξ) #ϵ learning rate, reps quali ripetizioni calcolare magnetizz
    σc=data
    maxrep=maximum(reps)
    M=[(it=r,m_v=copy(ξ[1,:])) for r in reps] #per ogni valore di reps, associo magnetizz risp P pattern
    iM=1
    if 0 ∈ reps
        M[iM]= (it=0, m_v=magnet(ξ,β,σc))
        iM+=1
    end
    for rep in 1:maxrep
        CDstep(ϵ,β,ξ,σc) #aggiorno ξ
        if rep ∈ reps #se è tra le ripetizioni fissate mantengo in memoria la magnetizzazione
            M[iM]= (it=rep, m_v=magnet(ξ,β,σc))
            iM+=1
        end
    end
    M
end


#test:
ϵ=0.01 #learnig rate
β=1/0.01 #1/temp
r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
N=100
P=3
#creazione dataset
ξq=rand((-1.0,1.0),N,P) #P patterns
p=0.2
tot_ex=2
data_rand=dataset(ξq, p, tot_ex)


ξ=mean(data_rand[ex,:,:] for ex in 1:tot_ex)
ξ=ξ'
#m_vv=zeros(tot_ex, P, size(r)[1])
m_vv=[]
#CD su ogni esempio
for ex in 1:tot_ex, μ in 1:P
    append!(m_vv,magn_mean(ϵ,r,β, data_rand[ex,μ,:], ξ)[:m_v])
end

#grafico 
#pygui(true)
#plot(r,m_vv[1+44*3:44*4])





