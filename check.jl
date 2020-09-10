
cd(@__DIR__)
include("hbm.jl")

using PyPlot 
#using Statistics

using LinearAlgebra
using Statistics

using ProgressMeter


function magnet(ξ, β, σ)
    N,P =size(ξ)
    mean(begin
    (σ, z)=mctransition(β,ξ,σ)
    mag=((sign.(ξ)')*σ)./length(σ) # calcola magnetizzazione sign. serve per evitare problemi se xi non corrisponde a +/- 1
    mag=@.(mag*sign(σ[1]*ξ[1,:])) # rompo l'invarianza di gauge della magnetizzazione per poter mediare osservazioni indipendenti
    end for i in 1:30)
end

function test(ϵ,reps,N,P, β) #ϵ learning rate, reps quali ripetizioni calcolare magnetizz
    ξ=rand((-1.0,1.0),N,P) #P patterns
    σc=ξ[:,1] #esempio con un dato rumore q 
    q=0.2
    flip=floor(Int, q*N) #numero di spin da flippare
    for i in 1:flip
        s=rand((1:N)) #scelgo random quali spin flippare
        σc[s]=-σc[s]
    end
    maxrep=maximum(reps)
    M=[(it=r,m_v=copy(ξ[1,:])) for r in reps] #per ogni valore di reps, associo magnetizz risp P pattern
    ξv=copy(ξ)
    iM=1
    if 0 ∈ reps
        M[iM]= (it=0, m_v=magnet(ξ,β,σc))
        iM+=1
    end
    for rep in 1:maxrep
        ξv=CDstep(ϵ,β,ξ,σc) #aggiorno ξ
        ξv=(ξv.-mean(ξv))./std(ξv)
        if rep ∈ reps #se è tra le ripetizioni fissate mantengo in memoria la magnetizzazione
            M[iM]= (it=rep, m_v=magnet(ξv,β,σc))
            iM+=1
        end
    end
    #print(mean(ξv))
    #print("\n")
    #print(std(ξv))
    #print("\n")
    M
end


function magn_mean(ϵ,r,N,P, β) #calcolo media delle magnetizzazione trovate nel test
    runs=20
    V=@showprogress [test(ϵ,r,N,P, β) for i in 1:runs]
    (reps=r,
    m_v=mean(getfield.(v,:m_v) for v in V)) #getfield da v prende solo i valori contrassegnati m_v
end

#test:
ϵ=0.01
β=10
r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
N=100
P=3
m_v=magn_mean(ϵ,r,N,P,β)[:m_v]

#fattore 10 che non va? per beta=1/10 paramagnetico

#plot magnetizzazione
pygui(true)
plot(r,m_v[:])



#plot della magn risp pattern da cui proviene l'esempio
#aa=zeros(size(r)[1])
#for i in 1:size(r)[1]
#    aa[i]=m_v[i][1]
#end
#pygui(true)
#plot(r,aa)


#test con β differenti - da controllare
#ϵ=0.01
#βv=1 ./(0.1:0.4:2)
#r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
#N=100
#P=3
#aa=zeros((size(βv)[1],size(r)[1]))
#for iM in 1:size(βv)[1]
#    β=βv[iM]
#    m_v=magn_mean(ϵ,r,N,P,β)[:m_v]
#    for i in 1:size(r)[1]
#        aa[iM]=m_v[i][1]
#    end
#end
#mass=maximum(aa, dims=1)