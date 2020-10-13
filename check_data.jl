cd(@__DIR__)
include("hbm.jl")

using PyPlot 
#using Statistics

using LinearAlgebra
using Statistics

using ProgressMeter


function dataset(ξ, q, tot_ex) #creazione del dataset 
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

function magn_mean(ϵ,r, β, data, ξ) #calcolo media delle magnetizzazione trovate nel test
    runs=30
    V=@showprogress [test(ϵ,r, β, data, ξ) for i in 1:runs]
    (reps=r,
    m_v=mean(getfield.(v,:m_v) for v in V), ξ_n=[getfield(V[i][length(r)], :ξ_n) for i in 1:runs]) #getfield da v prende solo i valori contrassegnati m_v
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
    M=[(it=r,m_v=copy(ξ[1,:]), ξ_n=copy(ξ)) for r in reps] #per ogni valore di reps, associo magnetizz risp P pattern
    ξv=copy(ξ)
    iM=1
    if 0 ∈ reps
        M[iM]= (it=0, m_v=magnet(ξv,β,σc), ξ_n=ξv)
        iM+=1
    end
    for rep in 1:maxrep
        ξv=CDstep(ϵ,β,ξ,σc) #aggiorno ξ
        ξv= (ξv.-mean(ξv))./std(ξv)
        if rep ∈ reps #se è tra le ripetizioni fissate mantengo in memoria la magnetizzazione
            M[iM]= (it=rep, m_v=magnet(ξv,β,σc), ξ_n=ξv)
            iM+=1
        end
    end
    M
end

function magn_mean_data(ϵ, r, β, data, ξn )
    N,P=size(ξn)
    tot_ex=size(data)[1]
    m_vv=[]
    #o_vv=[]
    #CD su ogni esempio
    for ex in 1:tot_ex
        for μ in 1:P
            V=magn_mean(ϵ,r,β, data[ex,μ,:], ξn)
            append!(m_vv, [V[:m_v][size(r)[1]]])
            #append!(o_vv, [V[:o_v]])
            ξn=last(V[:ξ_n])
        end
    end
    (m_vv=m_vv, ξ_n=ξn)
end

#test:
ϵ=0.01 #learnig rate
β= 1/0.5 #1/temp
r=[0; trunc.(Int,floor.(1.15.^(1:50))|>unique)]
N=100
P=4
#creazione dataset
ξq=rand((-1.0,1.0),N,P) #P patterns

#ξq=rand(Binomial(1,0.5),(N,P))
#ξq=ifelse.(ξq .== 1, 1, -1)

p=0.2
tot_ex1=500
data_rand=dataset(ξq, p, tot_ex1)

ξ=reshape(mean(data_rand, dims=1), (P,N))#mean(data_rand[ex,:,:] for ex in 1:tot_ex1) #media degli esempi
ξ=ξ'


tot_ex=10
V=magn_mean_data(ϵ, r, β, data_rand[1:tot_ex,:,:], ξ)

m_vv=V[:m_vv]

data_rand2=reshape(data_rand, (P*tot_ex1, N))
function statistics2(thresh, data_rand2, m_vv)
    good, wrong, spur=[],[],[]
    good_ex, wrong_ex, spur_ex= [],[],[]
    for tot in 0:P*tot_ex-1
        example=data_rand2[tot+1,:]
        leng1=length(good)
        leng2=length(wrong)
        for ν in 1:P
            if (m_vv[tot+1][ν] - (10^(-4)) >= thresh) && (ν == mod(tot,P)+1)
                append!(good,1)
                append!(good_ex, [example])
            elseif (m_vv[tot+1][ν] - (10^(-4)) >= thresh) && length(good)==leng1
                append!(wrong,1)
                append!(wrong_ex, [example]) 
            end
        end
        if length(good)==leng1 && length(wrong) == leng2
                append!(spur,1)
                append!(spur_ex, [example])
        end
    end
    length(good), length(wrong), length(spur)#, good_ex, wrong_ex, spur_ex 
end
#good_num, wrong_num, spur_num =statistics2(0.85,data_rand2[1:P*tot_ex, :], m_vv)


function statistics1(thresh, data_rand2, m_vv)
    good, wrong, spur=[],[],[]
    good_ex, wrong_ex, spur_ex= [],[],[]
    for tot in 0:P*tot_ex-1
        example=data_rand2[tot+1,:]
            if findmax(m_vv[tot+1])[1] > thresh
                if (findmax(m_vv[tot+1])[2] == mod(tot,P)+1)
                 append!(good,1)
                    append!(good_ex, [example])
                else
                    append!(wrong,1)
                    append!(wrong_ex, [example]) 
                end
            else 
                append!(spur,1)
                append!(spur_ex, [example])
            end
    end
    length(good), length(wrong), length(spur)#, good_ex, wrong_ex, spur_ex 
end
good_num, wrong_num, spur_num =statistics1(0.85,data_rand2[1:P*tot_ex, :], m_vv)

pygui(true)
matshow(overlap_weigths(ξ, V[:ξ_n]))
plt.title("Last overlap matrix weigths: β="*string(β)*", α="*string(P/N)*"\ngood= "*string(good_num)*"wrong="*string(wrong_num))
colorbar()

magnn=[]
for ex in 1:tot_ex
    for μ in 1:P
        append!(magnn, [magnet(V[:ξ_n], β, data_rand[ex, μ, :])])
    end
end
