## This generates figures and tables that might be used in the paper.

##
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using BSON: @load
using Statistics, MCMCChains, StatsPlots, DataFrames, Term, KernelDensity

function main()

## READ IN 30-20 CUE
@info "reading chains for 30-20-CUE"
files = ("30-20-CUE-chain1.bson", "30-20-CUE-chain2.bson", "30-20-CUE-chain3.bson")
chain1 = nothing
for f in files
    @load f chain Σp
    chain1 == nothing ? chain1 = chain : chain1 = [chain1; chain]
    println("rows: ", size(chain1,1))
end
println("30-20-CUE loaded, acceptance rate: ", mean(chain1[:,end-1]), " length: ", size(chain1,1))

## READ IN 30-20-16-19 CUE
@info "reading chains for 30-20-16-19-CUE"
files = ("30-20-16-19-CUE-chain1.bson",)
chain2 = nothing
for f in files
    @load f chain Σp
    chain2 == nothing ? chain2 = chain : chain2 = [chain2; chain]
    println("rows: ", size(chain2,1))
end
println("30-20-16-19 CUE loaded, acceptance rate: ", mean(chain2[:,end-1]), " length: ", size(chain2,1))


## make jump size zero when there are no jumps
chain1[:,7] = chain1[:,7] .* (chain1[:,6] .> 0.0)
chain2[:,7] = chain2[:,7] .* (chain2[:,6] .> 0.0)

## summaries of chains, and quantiles
dfnames = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ","ac","lnL"]   
println(@green "Chain summary for 14-17 data")
display(Chains(chain1, dfnames))
println(@green "Chain summary for 16-19 data")
display(Chains(chain2, dfnames))

## make data frames for plots, etc
spy1 = DataFrame(chain1, dfnames)
spy2 = DataFrame(chain2, dfnames)
println(@green "additional summary for 14-17 data")
@show describe(spy1)
println(@green "additional summary for 16-19 data")
@show describe(spy2)

## make plots
bandwidth = 0.01
for i = 1:8
    density(spy1[:,i],fill=true, alpha=0.5, label="2014-2017")
    p = density!(spy2[:,i], fill=true, alpha=0.5, title=dfnames[i], label="2016-2019")
    #savefig(dfnames[i]*".png") # already saved, uncomment to overwrite
    # uncomment following to see interactively
    display(p)
    sleep(5) # for interactive
end

## look at bivariate density contours
for i = 1:7
    for j = i+1:8
        p = contour(kde((spy1[:,i], spy1[:,j])), xlabel=dfnames[i], ylabel=dfnames[j])
        # already saved, uncomment following to overwrite
        #savefig(dfnames[i]*dfnames[j]*".png")
        # uncomment to see interactively
        display(p)
        sleep(5)
        
    end
end    

##

end
main()