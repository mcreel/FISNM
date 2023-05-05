using Pkg
Pkg.activate(".")
using BSON: @load
using Statistics, MCMCChains, StatsPlots

function main()

#files = ("chain1_rv30_cap0.bson", "chain2_rv30_cap0.bson", "chain3_rv30_cap0.bson","chain4_rv30_cap0.bson")
files = ("chain4_rv30_cap0.bson")

#files = ("chain1_rv30_capNone.bson", "chain2_rv30_capNone.bson", "chain3_rv30_capNone.bson")
ch = nothing
Σp = 1.

for f in files
    @load f chain Σp
    ch == nothing ? ch = chain : ch = [ch; chain]
    println("rows: ", size(ch,1))
end

@info "acceptance rate: " mean(ch[:,end])
ch = ch[:,1:end-1]

names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ"]   
ch = Chains(ch, names)

display(ch)
display(plot(ch))

end
main()
