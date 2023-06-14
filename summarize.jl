using Pkg
Pkg.activate(".")
using BSON: @load
using Statistics, MCMCChains, StatsPlots

function main()
files = ("30-20-CUE-chain1.bson", "30-20-CUE-chain2.bson")

ch = nothing
Σp = 1.
for f in files
    @load f chain Σp
    ch == nothing ? ch = chain : ch = [ch; chain]
    println("rows: ", size(ch,1))
end

@info "acceptance rate: " mean(ch[:,end-1])

# make jump size zero if there are no jumps
ch[:,7] = ch[:,7] .* (ch[:,6] .> 0.0)
names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ","ac","lnL"]   
ch2 = Chains(ch, names)

display(ch2)
display(plot(ch2))
#=
plots = Any[]
for i = 1:7
    for j = i+1:8
        push!(plots, marginalkde(ch[:,i], ch[:,j], xlabel=names[i], ylabel=names[j]))
    end
end    

plots
=#
nothing
end
main()
