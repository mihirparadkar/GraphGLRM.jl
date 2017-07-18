#Evaluates the loss function over all the observed values.
function threaded_loss_objective{T <: AbstractMatrix{Float64}}(g::GGLRM, XY::T)
  yidxs = get_yidxs(g.losses)
  obj = zeros(Threads.nthreads())
  Threads.@threads for j in 1:length(g.losses)
    obsex = g.observed_examples[j]
    #A may be a DataFrame or DataArray, but native Arrays are fastest
    @inbounds Aj = convert(Array, g.A[obsex, j])
    #To allow the sparse version to work as well
    @inbounds XYj = convert(Array, XY[obsex, yidxs[j]])
    obj[Threads.threadid()] += evaluate(g.losses[j], XYj, Aj)
  end
  sum(obj)
end

#Makes some performance optimizations
#Finds all of the gradients in a column so that the size of the column gradient is consistent
#as opposed to the row gradient which has column-chunks and whatnot
@inline function _threadedupdateGradX!{T <: AbstractMatrix{Float64}}(g::AbstractGLRM, XY::T, gx::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  scale!(gx,0)

  #Update the gradient, go by column then by row
  Threads.@threads for j in 1:length(g.losses)
    #Yj for computing gradient
    @inbounds Yj = view(g.Y, :, yidxs[j])
    obsex = g.observed_examples[j]

    #Take whole columns of XY and A and take the gradient of those
    @inbounds Aj = convert(Array, g.A[obsex, j])
    @inbounds XYj = convert(Array, XY[obsex, yidxs[j]])
    grads = grad(g.losses[j], XYj, Aj)

    #Single dimensional losses
    if isa(grads, Vector)
      for e in 1:length(obsex)
        #i = obsex[e], so update that portion of gx
        @inbounds gxi = view(gx, :, obsex[e])
        axpy!(grads[e], Yj, gxi)
      end
    else
      for e in 1:length(obsex)
        @inbounds gxi = view(gx, :, obsex[e])
        gemm!('N','N',1.0,Yj, grads[e,:], 1.0, gxi)
      end
    end
  end
end

@inline function _threadedupdateGradY!{T <: AbstractMatrix{Float64}}(g::AbstractGLRM, XY::T, gy::Matrix{Float64})
  yidxs = get_yidxs(g.losses)
  #scale y gradient to zero
  scale!(gy, 0)

  #Update the gradient
  Threads.@threads for j in 1:length(g.losses)
    @inbounds gyj = view(gy, :, yidxs[j])
    obsex = g.observed_examples[j]
    #Take whole columns of XY and A and take the gradient of those
    @inbounds Aj = convert(Array, g.A[obsex, j])
    @inbounds XYj = convert(Array, XY[obsex, yidxs[j]])
    grads = grad(g.losses[j], XYj, Aj)
    #Single dimensional losses
    if isa(grads, Vector)
      for e in 1:length(obsex)
        #i = obsex[e], so use that for Xi
        @inbounds Xi = view(g.X, :, obsex[e])
        axpy!(grads[e], Xi, gyj)
      end
    else
      for e in 1:length(obsex)
        @inbounds Xi = view(g.X, :, obsex[e])
        gemm!('N','T',1.0, Xi, grads[e,:], 1.0, gyj)
      end
    end
  end
end

#Does a line search for the step size for X, returns the new step size
@inline function _threadedproxStepX!(g::AbstractGLRM, params::ProxGradParams,
                            newX::Matrix{Float64}, gx::Matrix{Float64},
                            XY::Matrix{Float64}, newXY::Matrix{Float64},
                            αx::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_features))+1#(mapreduce(length,+,g.observed_features) + 1)

  obj = threaded_loss_objective(g,XY) + evaluate(g.rx, g.X)
  newobj = NaN
  while αx > params.min_stepsize #Linesearch to find the new step size
    stepsize = αx/l

    axpy!(-stepsize, gx, newX)
    prox!(g.rx, newX, stepsize)
    At_mul_B!(newXY, newX, g.Y)
    newobj = threaded_loss_objective(g, newXY) + evaluate(g.rx, newX)
    #newobj = threaded_objective(g, newXY)
    if newobj < obj
      copy!(g.X, newX)
      αx *= 1.03#(1.05)
      break
    else #Try again with smaller step-size
      copy!(newX, g.X)
      αx *= 0.9 #0.7
      if αx < params.min_stepsize
        αx = params.min_stepsize * 1.1
        break
      end #if
    end #if else
  end #while
  αx, newobj
end

@inline function _threadedproxStepY!(g::AbstractGLRM, params::ProxGradParams,
                              newY::Matrix{Float64}, gy::Matrix{Float64},
                              XY::Matrix{Float64}, newXY::Matrix{Float64},
                              αy::Number)
  #l = 1.5
  l = maximum(map(length, g.observed_examples)) + 1#(mapreduce(length,+,g.observed_features) + 1)
  #obj = threaded_objective(g,XY)
  obj = threaded_loss_objective(g, XY) + evaluate(g.ry, g.Y)
  newobj = NaN
  while αy > params.min_stepsize #Linesearch to find the new step size
    stepsize = αy/l
    axpy!(-stepsize, gy, newY)
    prox!(g.ry, newY, stepsize)
    At_mul_B!(newXY, g.X, newY)
    newobj = threaded_loss_objective(g, newXY) + evaluate(g.ry, newY)
    #newobj = threaded_objective(g, newXY)
    if newobj < obj
      copy!(g.Y, newY)
      αy *= 1.03#1.05
      break
    else #Try again with smaller step-size
      copy!(newY, g.Y)
      αy *= 0.9 #0.7
      if αy < params.min_stepsize
        αy = params.min_stepsize * 1.1
        break
      end #if
    end #if else
  end #while
  αy, newobj
end

function fit_multithread!(g::GGLRM;
                      params::ProxGradParams=ProxGradParams(),
                      ch::ConvergenceHistory=ConvergenceHistory("ProxGradGLRM"),
                      verbose=true)
  X,Y = g.X, g.Y
  A = g.A
  losses, rx, ry = g.losses, g.rx, g.ry

  #Initialize X*Y
  XY = At_mul_B(X,Y)

  tm = 0
  update_ch!(ch, tm, whole_objective(g,XY))
  (objx, objy) = (threaded_loss_objective(g, XY) + evaluate(g.rx, g.X)), (threaded_loss_objective(g, XY) + evaluate(g.ry, g.Y))
  #Step sizes
  αx = params.stepsize
  αy = params.stepsize

  # stopping criterion: stop when decrease in objective < tol, scaled by the number of observations
  scaled_abs_tol = params.abs_tol * mapreduce(length,+,g.observed_features)

  newX = copy(X)
  newY = copy(Y)
  newXY = copy(XY)

  #Gradient matrices
  gx = zeros(X)
  gy = zeros(Y)

  if verbose println("Fitting GGLRM") end
  for t in 1:params.max_iter
    #X update-----------------------------------------------------------------
    for xit in 1:params.inner_iter_X
      #_threadedupdateGradX!(g,XY,gx)
      _threadedupdateGradX!(g,XY,gx)
      #Take a prox step with line search
      αx, objx = _threadedproxStepX!(g, params, newX, gx, XY, newXY, αx)
      At_mul_B!(XY, X, Y) #Get the new XY matrix for objective
    end
    #Y Update---------------------------------------------------------------
    for yit in 1:params.inner_iter_Y
      #_threadedupdateGradY!(g,XY,gy)
      _threadedupdateGradY!(g,XY,gy)
      αy, objy = _threadedproxStepY!(g, params, newY, gy, XY, newXY, αy)
      At_mul_B!(XY, X, Y) #Get the new XY matrix for objective
    end
    obj = threaded_loss_objective(g, XY) + evaluate(g.rx, g.X) + evaluate(g.ry, g.Y)
    if t % 10 == 0
      if verbose
        println("Iteration $t, objective value: $(obj)")
      end
    end
    #Update convergence history
    tm = time() - tm
    update_ch!(ch, tm, obj)
    tm = time()
    #Check stopping criterion
    obj_decrease = ch.objective[end-1] - obj
    if t>10 && (obj_decrease < scaled_abs_tol || obj_decrease/obj < params.rel_tol)
      if verbose
        println("Iteration $t, objective value: $obj")
      end
      break
    end
  end #For
  ch
end
