module AutoOptimize

using DiffEqBase, ModelingToolkit, SparsityDetection, SparseDiffTools,
      CUDA, Logging, LinearAlgebra, SparseArrays, OrdinaryDiffEq, StaticArrays

MULTITHREADING_CUTOFF = 2^10
SPARSE_CUTOFF = 2^8
SPARSE_PERCENTAGE_CUTOFF = 0.01
COLORVEC_PERCENTAGE_CUTOFF = 0.5
STATIC_ARRAY_CUTOFF = 10

"""
tune_system

Tunes SciML variables in auto_optimize and beyond to maximize performance.
"""
function tune_system()

end

"""
auto_optimize(prob,alg=nothing;kwargs...)

A go fast button for those who are lazy or just don't know how to
optimize code. I don't trust that your code is fast, you don't trust
that your code is fast, so let's just get this over with and call it
a day.
"""
function auto_optimize(prob::ODEProblem,alg=nothing;
                       verbose = true,
                       stiff = true,
                       mtkify = true,
                       sparsify = true,
                       gpuify = true,
                       static = true,
                       gpup = nothing)
      N = length(prob.u0)

      if mtkify
            verbose && println("Try ModelingToolkitization")
            try
                  sys = modelingtoolkitize(prob)
                  jac = calculate_jacobian(sys,sparse=false)
                  sparsejac = SparseArrays.sparse(jac)
                  sparsity_percentage = length(nonzeros(sparsejac))/length(vec(jac))

                  if N > SPARSE_CUTOFF && sparsity_percentage < SPARSE_PERCENTAGE_CUTOFF
                        sys.jac[] = sparsejac
                  end

                  form = N > MULTITHREADING_CUTOFF ? ModelingToolkit.MultithreadedForm() : ModelingToolkit.SerialForm()
                  static = static && N < STATIC_ARRAY_CUTOFF

                  if static
                        prob = ODEProblem{false}(sys,SArray{Tuple{size(prob.u0)...}}(prob.u0),prob.tspan,prob.p,
                                          jac = true, tgrad = true, simplify = true,
                                          sparse = false,
                                          parallel = false,
                                          prob.kwargs...)
                  else
                        prob = ODEProblem(sys,prob.u0,prob.tspan,prob.p,
                                          jac = true, tgrad = true, simplify = true,
                                          sparse = N > SPARSE_CUTOFF &&
                                                   sparsity_percentage < SPARSE_PERCENTAGE_CUTOFF,
                                          parallel = form,
                                          prob.kwargs...)
                  end
                  return prob,alg
            catch e
                  @warn("ModelingToolkitization Approach Failed")
                  verbose && println(e)
                  throw(e)
            end
      end

      if stiff && N > SPARSE_CUTOFF && sparsify
            verbose && println("Try SparsityDetection")
            try
                  input = copy(prob.u0)
                  output = similar(input)
                  function f(du,u,p,t)
                        if isinplace(prob.f)
                              prob.f.f(du,u,p,t)
                        else
                              du .= prob.f.f(u,p,t)
                        end
                        return nothing
                  end
                  sparsity_pattern = jacobian_sparsity(f,output,input,SparsityDetection.Fixed(prob.p),prob.tspan[1])
                  sparsejac = sparse(sparsity_pattern)
                  sparsity_percentage = length(nonzeros(sparsejac))/prod(size(jac))
                  if sparsity_percentage < SPARSE_PERCENTAGE_CUTOFF && sparsity_percentage > COLORVEC_PERCENTAGE_CUTOFF
                        # Doesn't make sense to do sparsity for lu but does make sense for differentiation
                        colorvec = matrix_colors(sparsejac)
                        _f = ODEFunction(prob.f, jac=prob.jac, sparsity = sparsejac, tgrad = prob.tgrad, colorvec=colorvec)
                        prob = remake(prob;f=_f)
                  elseif sparsity_percentage > SPARSE_PERCENTAGE_CUTOFF
                        colorvec = matrix_colors(sparsejac)
                        _f = ODEFunction(prob.f, jac=prob.jac, jac_prototype = sparsejac, tgrad = prob.tgrad, colorvec=colorvec)
                        prob = remake(prob;f=_f)
                  else
                        error("Not sparse enough. Sparsity percentage = $sparsity_percentage")
                  end
            catch e
                  @warn("SparsityDetection Approach Failed")
                  verbose && println(e)
            end
      end

      if gpuify
            verbose && println("Try GPUification")
            try
                  CUDA.allowscalar(false)
                  u0 = prob.u0
                  gu0 = cu(u0)

                  if gpup isa Bool && gpup
                        gp = cu(prob.p)
                  elseif gpup isa Bool && !gpup
                        gp = prob.p
                  else
                        # Guess whether p should be on the GPU too
                        gp = typeof(prob.p) <: Array && eltype(prob.p) <: AbstractFloat && length(prob.p) > 100 ? cu(prob.p) : prob.p
                  end


                  if DiffEqBase.isinplace(prob)
                        gdu0 = similar(gu0)
                        prob.f(gdu0,gu0,gp,prob.tspan[1])
                        gputime = @elapsed prob.f(gdu0,gu0,gp,prob.tspan[1])
                  else
                        prob.f(gu0,gp,prob.tspan[1])
                        gputime = @elapsed prob.f(gu0,gp,prob.tspan[1])
                  end

                  if DiffEqBase.isinplace(prob)
                        du0 = similar(u0)
                        prob.f(du0,u0,prob.p,prob.tspan[1])
                        cputime = @elapsed prob.f(du0,u0,prob.p,prob.tspan[1])
                  else
                        prob.f(u0,prob.p,prob.tspan[1])
                        cputime = @elapsed prob.f(u0,prob.p,prob.tspan[1])
                  end

                  cputime < gputime/2 && error("GPU did not increase `f` speed")
                  prob = remake(prob,u0=gu0,p=gp)
            catch e
                  @warn("GPUificiation Approach Failed")
                  verbose && println(e)
            end
      end
      prob,alg
end

"""
auto_optimize_multiple(prob,alg=nothing;kwargs...)

Look, you're going to be doing a bunch of parameter estimation and
stuff on this, so you need to solve this multiple times, and by
multiple I mean 10,000x. So by all means, go ahead and call this and
I'll use a few solves to figure out something that will hopefully
make it go zoooooooooom.
"""
function auto_optimize_multiple(prob::ODEProblem,alg=nothing;kwargs...)
      prob,alg
end

export tune_system, auto_optimize, auto_optimize_multiple

end
