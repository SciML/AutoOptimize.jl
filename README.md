# AutoOptimize

Do you want your calculation to go faster but are also too lazy to
optimize your code? Well I have a solution for you! Introducing the
AutoOptimize system. With a simple `]add AutoOptimize`, we'll send
it straight to your home and SHIPPING IS FREE! In seconds your
calculations will be taking seconds. So what do you have to lose?
Here, come on in and I'll show you how to use it.

## Making Code Fast, the Lazy Way

Define an ODEProblem, `prob`. Alright, I'll wait for you to catch
up. Now call:

```julia
_prob = auto_optimize(prob)
```

Now `_prob` is better, so go use that one.

## Example: Faster PDEs

```julia
using AutoOptimize, OrdinaryDiffEq, LinearAlgebra, SparseArrays

# Define the constants for the PDE
const α₂ = 1.0
const α₃ = 1.0
const β₁ = 1.0
const β₂ = 1.0
const β₃ = 1.0
const r₁ = 1.0
const r₂ = 1.0
const _DD = 100.0
const γ₁ = 0.1
const γ₂ = 0.1
const γ₃ = 0.1
const N = 32
const X = reshape([i for i in 1:N for j in 1:N],N,N)
const Y = reshape([j for i in 1:N for j in 1:N],N,N)
const α₁ = 1.0.*(X.>=4*N/5)

const Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
const My = copy(Mx)
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0

# Define the discretized PDE as an ODE function
function f(u,p,t)
    A = @view  u[:,:,1]
    B = @view  u[:,:,2]
    C = @view  u[:,:,3]
    MyA = My*A
    AMx = A*Mx
    DA = @. _DD*(MyA + AMx)
    dA = @. DA + α₁ - β₁*A - r₁*A*B + r₂*C
    dB = @. α₂ - β₂*B - r₁*A*B + r₂*C
    dC = @. α₃ - β₃*C + r₁*A*B - r₂*C
    cat(dA,dB,dC,dims=3)
end

u0 = zeros(N,N,3)
MyA = zeros(N,N);
AMx = zeros(N,N);
DA = zeros(N,N);
prob = ODEProblem(f,u0,(0.0,10.0))
```

But then you're like "but diz code is beautiful!" and I'm just proper
chuffed: how do I make my benchmarks look good if you don't want to
write good code?

Well, I guess this calls for Doctor Auto Optimize!

```julia
_prob,_alg = auto_optimize(prob)
```

After this churns away for a bit, you go boom:

```julia
@btime solve(_prob, TRBDF2()) # 168.558 ms (4925 allocations: 101.41 MiB)
```

and there you go, now you're solving 4 PDEs a second. What was it like
before the optimization?

```julia
@btime solve(prob, TRBDF2(autodiff=false)) # 249.993 s (18560715 allocations: 1281.93 GiB)
```

That's a lean 1483x temporal speedup and 12945x reduction in memory requirements!
