using Pkg
Pkg.activate(".")
using Statistics, Flux, PrettyTables
using Base.Iterators

include("MA2lib.jl")

@views function lags(x,p)
    cols = size(x,2)
    result = zeros(size(x,1), cols*p)
    for i = 1:p
        result[:,i*cols-cols+1:i*cols] = [ones(i,cols); x[1:end-i,:]]
    end
    return result
end

# fits an AR(P) model to MA(2) data
@views function auxstat(theta, n, P)
    y = ma2(theta,n)
    x = lags(y,P)
    y = y[P+1:end]
    x = [ones(n-P,1) x[P+1:end,:]]
    Z = x\y
    return Z
end    

function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data .= max.(data, q01)
    data .= min.(data, q99)
    data .= (data .- q50) ./ iqr
end

@views function MakeNeuralMoments(params, statistics, trainsize, testsize; Epochs=1000, verbose=false)
    nParams = size(params,1)
    nStats = size(statistics,1)
    # transform stats to robustify against outliers
    q50 = zeros(nStats)
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:nStats
        q = quantile(statistics[i,:],[0.01, 0.25, 0.5, 0.75, 0.99])
        q01[i] = q[1]
        q50[i] = q[3]
        q99[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    nninfo = (q01, q50, q99, iqr) 
    for i = 1: nStats
        statistics[:,i] .= TransformStats(statistics[:,i], nninfo)
    end    
    # train net
    params = Float32.(params)
    s = Float32.(std(params, dims=2))
    statistics = Float32.(statistics)
    # training, testing, and final monte carlo fit
    yin = params[:,1:trainsize]
    yout = params[:,trainsize+1:trainsize+testsize]
    ymc = params[:,trainsize+testsize+1:end]
    xin = statistics[:,1:trainsize]
    xout = statistics[:,trainsize+1:trainsize+testsize]
    xmc = statistics[:,trainsize+testsize+1:end]
    # define the neural net
    NNmodel = Chain(
        Dense(nStats, 10*nParams, tanh),
        Dense(10*nParams, 3*nParams, tanh),
        Dense(3*nParams, nParams)
    )
    loss(x,y) = Flux.huber_loss(NNmodel(x)./s, y./s; δ=0.1) # Define the loss function
    # monitor training
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    # do the training
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 50)]
    bestmodel = 0.0
    for i = 1:Epochs
        if i < 20
            opt = Momentum() # the optimizer
        else
            opt = ADAM() # the optimizer
        end 
        Flux.train!(loss, Flux.params(NNmodel), batches, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            bestmodel = NNmodel
            if verbose
                xx = xout
                yy = yout
                println("________________________________________________________________________________________________")
                monitor(i)
                pred = NNmodel(xx)
                error = yy .- pred
                results = [pred;error]
                rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
                println(" ")
                println("testing RMSE for model parameters ")
                pretty_table(reshape(round.(rmse,digits=3),1,nParams))
            end
        end
    end
    err = NNmodel(xmc) - ymc
    bestmodel, err
end

function main()
    rmaes = zeros(10)
    for rep = 1:10
        # sample size
        ns = (100)
        results = zeros(1,5)
        for n ∈ ns
            P = 10 # order of AR fit to the data
            trainsize = Int64(1e6)
            testsize = Int64(1e4)
            mcreps = 5000
            R = trainsize+testsize+mcreps
            Zs = zeros(P+1, R)
            thetas = PriorDraw(R)
            for i = 1:R
                Zs[:,i] = auxstat(thetas[:,i], n, P)
            end
            bestmodel, err = MakeNeuralMoments(thetas, Zs, trainsize, testsize; Epochs=100)
            mae = mean(abs.(err), dims=2)
            relmae = mae ./ [1.,0.5]
            rmaes[rep] = mean(relmae)
        end
        armae = mean(rmaes[1:rep])
        println("average E_% so far: $armae")
    end    
end
main()
