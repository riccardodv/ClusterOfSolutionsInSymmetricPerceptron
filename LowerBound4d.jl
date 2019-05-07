using HCubature
using LinearAlgebra
using SpecialFunctions
using QuadGK
using Optim
using LineSearches
using JuMP
using Ipopt
using TimerOutputs # profile with: print_timer(); reset_timer!();

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

@timeit function f2(k, x, q)
    σ = [1 1-2*x q[1] q[2];
        1-2*x 1 q[3] q[4];
        q[1] q[3] 1 1-2*x;
        q[2] q[4] 1-2*x 1]
    # σ = my_id(σ)
    # Σ = σ
    Σ = Symmetric(σ) + 1e-10I
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
    
    @timeit "hcub" hcubature(y-> begin
        yΣi4 = y⋅Σi4
        yΣi3y = y⋅(Σi3*y)
        return 1/((2π)^(3/2)*√Σd*sqi44)*
            exp(-0.5*yΣi3y + 0.5*yΣi4^2/sqi44^2)*
            (H(-sqi44*k+yΣi4/sqi44)
            -H(+sqi44*k+yΣi4/sqi44))
    end, -k*ones(3), k*ones(3), rtol=1e-5, maxevals=10^5)[1]
end

@timeit function f1(k, x)
    quadgk(y -> G(y)*(H((-k-(1-2*x)*y)/(2*√(x*(1-x))))-H((k-(1-2x)*y)/(2*√(x*(1-x))))), -k, k,
            rtol=1e-7, maxevals=10^7)[1]
end

function H8(x, η, q...)
    δ = 1e-10
    a = [1/4*(q[1]-q[2]+2*x)-η, 1/4*(-q[3]+q[4]+2*x)-η, 1/4*(2-q[1]-q[4]-4*x)+η,
            1/4*(q[1]-q[3]+2*x)-η, η]
    a0 = 1-a[1]-a[2]-a[3]-x
    a6 = x-a[1]-a[2]-a[5]
    a7 = a[1]+a[2]-a[4]
    [a[i] = (-δ < a[i] < 0) ?  0. : a[i] for i in 1:5]
    a0 = (-δ < a0 < 0) ?  0. : a0
    a6 = (-δ < a6 < 0) ?  0. : a6
    a7 = (-δ < a7 < 0) ?  0. : a7
    # println("a = $a\ta0 = $a0\ta6 = $a6\ta7 = $a7")
    for i in 1:5 @assert 0<=a[i]<=1. end
    @assert 0<=a0<=1.
    @assert 0<=a6<=1.
    @assert 0<=a7<=1.
    shannon_entropy = sum(-xlogx(aa) for aa in a)
    shannon_entropy += -xlogx(a0) - xlogx(a6) - xlogx(a7)
end


@timeit function compute_supH8(x, q, minB, maxA)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    @NLparameter(model, xx == x)
    @NLparameter(model, qq[i=1:4] == q[i])
    register(model, :xlogx, 1, xlogx, autodiff=true)
    register(model, :H8, 6, H8, autodiff=true)
    
    η = @variable(model, base_name="η", lower_bound=maxA, upper_bound=minB)
    set_start_value(η, (maxA+minB)/2)
    @NLobjective(model, Max, H8(xx, η, qq...))
    optimize!(model)
    return  objective_value(model)
end

function αLB(h2, logf1, supH8, k, x, q)
    (log(2)+2*h2-supH8)/(log(f2(k, x, q))-2*logf1)
end

@timeit function αLB(k, x, q)
    ok, maxA, minB = is_good(k, x, q)
    !ok && return ok, Inf

    f1val = f1(k, x)
    f2val = f2(k, x, q)
    f2val - f1val^2 < 1e-5 && return false, Inf # CHECK This
     
    if abs(minB-maxA) < 1e-5
        supH8 = H8(x, (maxA+minB)/2, q...)
    else
        supH8 = compute_supH8(x, q, minB, maxA)
    end
    h2 = H2(x)
    return ok,  (log(2)+2*h2-supH8)/(log(f2val)-2*log(f1val))
end

@timeit function is_good(k, x, q)
    ok = false
    maxA = max(1/4*(q[1]-q[2]+2*x-4), 1/4*(-4-q[3]+q[4]+2*x), 1/4*(-2+q[1]+q[4]+4*x),
                1/4*(-4+q[1]-q[3]+2*x), 0., 1/4*(q[1]-q[2]-q[3]+q[4]),
                1/4*(-4-q[2]+q[4]+2*x), 1/4*(-2-q[2]-q[3]+4*x))
    minB = min(1/4*(q[1]-q[2]+2*x), 1/4*(-q[3]+q[4]+2*x), 1/4*(2+q[1]+q[4]+4*x),
                1/4*(q[1]-q[3]+2*x), 1., 1/4*(q[1]-q[2]-q[3]+q[4]+4*x),
                1/4*(-q[2]+q[4]+2*x), 1/4*(2-q[2]-q[3]))

    ok =  maxA <= minB
    return ok, maxA, minB
end

@timeit function create_grid(n)
    return (q ./ n for q in Iterators.product(0:n, 0:n, 0:n, 0:n))
end

@timeit function optimize_on_grid(n, k, x, ϵ = 1e-5)
    GG = create_grid(n)
    grid_vals = []
    oks = 0
    for q in GG
        ok, alb = αLB(k, x, q) 
        if ok 
            push!(grid_vals, [q, 0, alb])
            println("q=$q  alb=$alb")
        end
        oks += ok
    end
    return oks, oks/n^4, grid_vals
end

## USAGE:
# oks, frac_oks, grid_vals = optimize_on_grid(20, 1., 0.01)
# display(grid_vals)
# print_timer(); reset_timer!();
# grid_vals[findmin([x[3] for x in grid_vals])[2]]

# The Following code is to debug testing the optimization on specific points q
# x = 0.1
# k = 1.
# q = [0.1, 0.1, 0.1, 0.1]
# #q = [0., 0.2, 0.2, 0.]
# ϵ = 1e-5
# ok, maxA, minB = is_good(k, x, q)
# print("che cazzo è?")
# println("maxA = $maxA, minB = $minB")
# if abs(maxA-minB)<ϵ
#     println("ECCEZIONE")
#     supH8 = H8(x, (maxA+minB)/2, q...)
#     h2 = H2(x); logf1 = log(f1(k, x))
#     αlb = αLB(h2, logf1, supH8, k, x, q)
#     println("q = $q\tln2h2 = $ln2h2\toptH8 = $optH8\th2 = $h2\tlogf1 = $logf1\tαlb = $αlb\tlogf2 = $logf2")
# end
# if ok == 1
#     model = Model(with_optimizer(Ipopt.Optimizer))
#     @NLparameter(model, xx == x)
#     η = @variable(model, base_name="η", lower_bound=maxA, upper_bound=minB)
#     set_start_value(η, (maxA+minB)/2)
#     @NLparameter(model, qq[i=1:4] == q[i])
#     register(model, :xlogx, 1, xlogx, autodiff=true)
#     register(model, :H8, 6, H8, autodiff=true)
#     @NLobjective(model, Max, H8(xx, η, qq...))
#     optimize!(model)
#     optH8 = objective_value(model)
#     h2 = H2(x); logf1 = log(f1(k, x))
#     ln2h2 = log(2)+2*h2
#     logf2 = log(f2(k, x, q))
#     αlb = αLB(h2, logf1, optH8, k, x, q)
#     println("q = $q\tln2h2 = $ln2h2\toptH8 = $optH8\th2 = $h2\tlogf1 = $logf1\tαlb = $αlb\tlogf2 = $logf2")
# else
#     println("Point q not accepted!")
# end
