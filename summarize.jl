using Pkg
Pkg.activate(".")
using BSON: @load
using Statistics, MCMCChains, StatsPlots

function main()

#files = ("chain1_rv30_cap0.bson", "chain2_rv30_cap0.bson", "chain3_rv30_cap0.bson","chain4_rv30_cap0.bson")
files = ("30-20-06-chain1.bson","30-20-06-chain2.bson")

ch = nothing
Σp = 1.

for f in files
    @load f chain Σp
    ch == nothing ? ch = chain : ch = [ch; chain]
    println("rows: ", size(ch,1))
end

@info "acceptance rate: " mean(ch[:,end-1])
#ch = ch[:,1:end-2]

names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ","ac","lnL"]   
ch = Chains(ch, names)

display(ch)
display(plot(ch))

end
main()
