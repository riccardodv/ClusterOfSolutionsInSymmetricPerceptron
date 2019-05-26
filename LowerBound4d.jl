using HCubature
using LinearAlgebra
using SpecialFunctions
using QuadGK
using Optim
using LineSearches
using JuMP
using Ipopt
using QuadGK
using DataFrames
using Gadfly

G(x) = exp(-x^2/2) / √(2π)
H(x) = erfc(x /√2) / 2
xlogx(x) = 0. < x <= 1. ? x*log(x) : 0
H2(x) = -xlogx(x)-xlogx(1-x)

## The functions my_inv and my_id are functions that I implemented to try to fix
## the problem of inverting matrices when an eigenvalue is too close to zero.
# function my_inv(M, ϵ = 1e-7)
#     d = eigvals(M)
#     [d[i] = (d[i] < ϵ) ? ϵ : d[i] for i in 1:4]
#     A = eigvecs(M)
#     dinv = d.^-1
#     A*Diagonal(dinv)*A'
# end
#
# function my_id(M, ϵ = 1e-7)
#     d = eigvals(M); A = eigvecs(M)
#     [d[i] = d[i] < ϵ ? ϵ : d[i] for i in 1:4]
#     A*Diagonal(d)*A'
# end
################################################################################

## This is a function for checks on the covariance matrix ######################
# function CovM(x, q)
#     σ = [1 1-2*x q[1] q[2];
#         1-2*x 1 q[3] q[4];
#         q[1] q[3] 1 1-2*x;
#         q[2] q[4] 1-2*x 1]
#     Σ = Symmetric(σ)
# end
################################################################################

function f2(k, x, q)
    σ = [1 1-2*x q[1] q[2];
        1-2*x 1 q[3] q[4];
        q[1] q[3] 1 1-2*x;
        q[2] q[4] 1-2*x 1]
    # σ = my_id(σ)
    # Σ = σ
    Σ = Symmetric(σ) + 1e-15I
    Σd = det(Σ)
    @assert all(eigvals(Σ) .> 0)
    Σi = inv(Σ)
    @assert Σi[4,4] > 0
    sqi44 = √Σi[4,4]
    # Σi = my_inv(Σ)
    Σi3 = Σi[1:3,1:3]
    Σi4 = Σi[1:3,4]
    ## Comment next three lines if everything works ############################
    # print("Σ = ")
    # display(Σ)
    # print("Σi = ")
    # display(Σi)
    # println(eigvals(Σ))
    # println("dΣ = $dΣ")
    ###########################################################################
    hcubature(y-> begin
        yΣi4 = y⋅Σi4
        yΣi3y = y⋅(Σi3*y)
        return 1/((2π)^(3/2)*√Σd*sqi44)*
            exp(-0.5*yΣi3y + 0.5*yΣi4^2/sqi44^2)*
            (H(-sqi44*k+yΣi4/sqi44)
            -H(+sqi44*k+yΣi4/sqi44))
    end, -k*ones(3), k*ones(3), atol=1e-5, maxevals=10^5)[1]
end

function f1(k, x)
    quadgk(y -> G(y)*(H((-k-(1-2*x)*y)/(2*√(x*(1-x))))-H((k-(1-2x)*y)/(2*√(x*(1-x))))),
                -k, k, rtol=1e-7, maxevals=10^7)[1]
end

function H8(x, η, q...)
    δ = 1e-7 # in the optimization in η it happens that ipopt goes below and above boundaries? seems so
    a = [1/4*(q[1]-q[2]+2*x)-η, 1/4*(-q[3]+q[4]+2*x)-η, 1/4*(2-q[1]-q[4]-4*x)+η,
            1/4*(q[1]-q[3]+2*x)-η, η]
    a0 = 1-a[1]-a[2]-a[3]-x
    a6 = x-a[1]-a[2]-a[5]
    a7 = a[1]+a[2]-a[4]
    [a[i] = (-δ < a[i] < 0) ?  0. : a[i] for i in 1:5]
    a0 = (-δ < a0 < 0) ?  0. : a0
    a6 = (-δ < a6 < 0) ?  0. : a6
    a7 = (-δ < a7 < 0) ?  0. : a7
    #println("a = $a\ta0 = $a0\ta6 = $a6\ta7 = $a7\tη = $η")
    #for i in 1:5 @assert 0<=a[i]<=1. end
    @assert 0<=a[1]<=1.; @assert 0<=a[2]<=1.; @assert 0<=a[3]<=1.;
    @assert 0<=a[4]<=1.; @assert 0<=a[5]<=1.;
    @assert 0<=a0<=1.
    @assert 0<=a6<=1.
    @assert 0<=a7<=1.
    shannon_entropy = sum(-xlogx(aa) for aa in a)
    shannon_entropy += -xlogx(a0) - xlogx(a6) - xlogx(a7)
end


function compute_supH8(x, q, minB, maxA)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    register(model, :xlogx, 1, xlogx, autodiff=true)
    register(model, :H8, 6, H8, autodiff=true)
    @NLparameter(model, xx == x)
    @NLparameter(model, qq[i=1:4] == q[i])
    η = @variable(model, base_name="η", lower_bound=maxA, upper_bound=minB)
    set_start_value(η, (maxA+minB)/2)
    @NLobjective(model, Max, H8(xx, η, qq...))
    optimize!(model)
    return  objective_value(model)
end

function αLB(k, x, q)
    ok, maxA, minB = is_good(k, x, q)
    #println("$maxA $minB")
    !ok && return ok, Inf

    f1val = f1(k, x)
    f2val = f2(k, x, q)
    #println("$(f1val^2) $f2val")
    #f2val - f1val^2 < 1e-8 && return false, Inf # CHECK This

    if abs(minB-maxA) < 1e-7
        supH8 = H8(x, (maxA+minB)/2, q...)
    else
        supH8 = compute_supH8(x, q, minB, maxA)
    end
    h2 = H2(x)
    return ok,  (log(2)+2*h2-supH8)/(log(f2val)-2*log(f1val))
end

function is_good(k, x, q)
    #ϵ = 1e-10
    ok = false
    maxA = max(1/4*(q[1]-q[2]+2*x-4), 1/4*(-4-q[3]+q[4]+2*x), 1/4*(-2+q[1]+q[4]+4*x),
                1/4*(-4+q[1]-q[3]+2*x), 0., 1/4*(q[1]-q[2]-q[3]+q[4]),
                1/4*(-4-q[2]+q[4]+2*x), 1/4*(-2-q[2]-q[3]+4*x))
    minB = min(1/4*(q[1]-q[2]+2*x), 1/4*(-q[3]+q[4]+2*x), 1/4*(2+q[1]+q[4]+4*x),
                1/4*(q[1]-q[3]+2*x), 1., 1/4*(q[1]-q[2]-q[3]+q[4]+4*x),
                1/4*(-q[2]+q[4]+2*x), 1/4*(2-q[2]-q[3]))
    #abs(maxA-minB)<ϵ && (minB=maxA)
    ok =  maxA <= minB
    return ok, maxA, minB
end

function create_grid(n)
    return (q ./ n for q in Iterators.product(0:n, 0:n, 0:n, 0:n))
end

function optimize_on_grid_LB(n, k, x)
    GG = create_grid(n)
    grid_vals = []
    oks = 0
    α_min = Inf
    q_min = [Inf, Inf, Inf, Inf]
    for q in GG
        ok, α_lb = αLB(k, x, q)
        if ok
            if α_lb < α_min
                α_min = α_lb
                q_min = q
            end
            push!(grid_vals, [q, α_lb])
            println("x=$x q=$q  α_lb=$α_lb")
        end
        oks += ok
    end
    return oks, oks/n^4, grid_vals, α_min, q_min
end

compute_UB(k, x) = -(log(2)+H2(x))/(log(f1(k, x)))

function compute_bounds(n, k)
    X = vcat([i for i in 1:9]*1e-3, [i for i in 1:9]*1e-2, [i for i in 1:5]*1e-1)
    # X = [i for i in 1:2:9]*1e-3
    bounds = []
    for x in X
        oks, frac, grid_vals, α_LB, q_LB = optimize_on_grid_LB(n, k, x)
        α_UB = compute_UB(k, x)
        push!(bounds, [x, q_LB, α_LB, α_UB])
    end
    return bounds
end

bounds = compute_bounds(10,1.)
bounds = hcat(bounds...)
display(bounds)
df = DataFrame(x = bounds[1,:], α_LB = bounds[3,:], α_UB = bounds[4,:])
plot = Gadfly.plot(df, x = :x, y = Col.value(:α_LB,:α_UB), color = Col.index(:α_LB,:α_UB))
plot |> PDF("bounds.pdf")

# some code to sudy the lower bound RS points
# k=1.; x = 0.01
# af1 = compute_UB(k,x)
# αLB(k, x, 0.99*ones(4))
# f2(k, x, 0.9989*ones(4))
# αLB(k, x, 0.998999*ones(4))
# is_good(k,x,0.9989*ones(4))
# H8(x,0.0005,0.9989*ones(4)...)
# q=0.981*ones(4)
# 1/4*(2-q[1]-q[4]-4*x)
#
# k=1. ; x = 0.01
# a=[]
# af1 = compute_UB(k,x)
# for q in 0.99:0.00001:0.9901
#     ok, aa = αLB(k, x, q .* ones(4))
#     global a = push!(a, [q, aa, af1])
#     println("$q $aa $af1")
# end
# b = hcat(a...)
# minimum(b[2,:])
# argmin(b[2,:])
# b[:,17]
