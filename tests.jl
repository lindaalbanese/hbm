cd(@__DIR__)
include("hbm.jl")


function sim(N,P,β)
    mean(begin
    ξ=sign.(randn(N,P))
    σ=ξ[:,1]
    ch=mcmc_chain(β, ξ, σ, 100)
    chain_estimators(ξ,ch;skip=10)[:M]
    end for i in 1:200)
end

using PyPlot
pygui(true)
P=3
for N = [25,50,100]
    βv=1 ./(0.1:0.03:4)
    S=sim.(N,P,βv)
    M=[getindex.(S,i) for i in 1:P]
    plot(1 ./ βv,M[1],label="\$m_{$N}\$")
    legend()
    sleep(0.05)
end
xlabel("T")
ylabel("M")
tight_layout()
savefig("fss.pdf")