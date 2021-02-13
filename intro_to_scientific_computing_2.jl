### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 0a02bfb2-6b1f-11eb-11f0-ef09f0118de2
using ForwardDiff

# ╔═╡ 93196008-6b1f-11eb-058a-79fca3e576a7
using TreeView

# ╔═╡ ef88c752-6b1f-11eb-1572-11c8caa573d1
using ModelingToolkit

# ╔═╡ 494f6d6e-6b24-11eb-11ce-efda4259cf0a
using PlutoUI

# ╔═╡ 76876386-6b1a-11eb-1038-974c75817b46
md"""
# Types in Julia
"""

# ╔═╡ 9a5e166a-6b1a-11eb-3e79-c90c8c86b312
md"""
**Type**: Label that tell how to manipulate data -- how an object **behaves**
"""

# ╔═╡ eee63e1a-6b1a-11eb-3566-43dd0ee7ba40
md"""
Example:
"""

# ╔═╡ f2629bd8-6b1a-11eb-3aef-37d264641bd1
(3, 4)

# ╔═╡ f9e00f9e-6b1a-11eb-272d-cf3787db796b
x = (3, 4)

# ╔═╡ fd22d768-6b1a-11eb-0d33-77830e60952e
Complex(3, 4)

# ╔═╡ 35260e00-6b1b-11eb-133d-f74b247b289f
[3, 4]

# ╔═╡ 391551c4-6b1b-11eb-1c91-bb491a24b056
md"""
## Automatic differentiation (forward mode) by defining new types
"""

# ╔═╡ 61083ff4-6b1b-11eb-38e9-85721910addc
md"""
Let's fix a point $a$, e.g. $a = 3$.

Want to calculate $f'(a)$
"""

# ╔═╡ 7b0098f2-6b1b-11eb-151b-1981d4b8bba1
md"""
Will calculate
$$f(a + \epsilon b) = f(a) + \epsilon * f'(b) + (1/2) * \epsilon^2 * f''(a) + \cdots$$
"""

# ╔═╡ b0737e08-6b1b-11eb-26e1-87348f0949aa
md"""
To order $\epsilon$:

$$f(a + \epsilon b) \simeq f(a) + \epsilon f'(b)$$
"""

# ╔═╡ c603da7e-6b1b-11eb-288a-098526cf1d31
md"""
$$(f + g)(a + \epsilon b) \simeq [f(a) + g(a)] + \epsilon [f'(a) + g'(a)] b$$
"""

# ╔═╡ 9a9e0d8e-6b1b-11eb-22f6-2d80328a81b0
md"""
Need type -- kind of object -- that stores two numbers and behaves according to these rules:

**Dual** number
"""

# ╔═╡ 094fa5d8-6b1c-11eb-10b2-3b9d703c3412
struct Dual
	value::Float64
	deriv::Float64
end

# ╔═╡ 34b8bd0e-6b1c-11eb-326b-47f1462c6f4e
methods(Dual)

# ╔═╡ 28f51046-6b1c-11eb-2f58-77f44bdb6146
z = Dual(1.0, 2.0)

# ╔═╡ 3f42684c-6b1c-11eb-272d-396ff9b9cb73
Dual(1, 2.0)

# ╔═╡ 7bd03f14-6b1c-11eb-06f3-2bc07ae05d66
md"""
`Dual(c, d)` is $c + \epsilon d$

`Dual(c, d)` represents some (any) function $f$ such that $f(a) = c$ and $f'(a) = d$
"""

# ╔═╡ 455878ac-6b1c-11eb-3adf-add5f4afcd84
z + z

# ╔═╡ 091f3a3c-6b1d-11eb-13dd-8bb44ae3e6a4
z

# ╔═╡ 0a211ec8-6b1d-11eb-3abc-9bb0a96f1e0b
z.value

# ╔═╡ 0fd33f2c-6b1d-11eb-1426-07abd3999e0a


# ╔═╡ 191a8b44-6b1d-11eb-0805-8bd851504811
z.value = 10

# ╔═╡ 5a68c3f0-6b1c-11eb-0a36-33a9f12539be
Base.:+( x::Dual, y::Dual) = Dual(x.value + y.value, x.deriv + y.deriv)

# ╔═╡ 4092d352-6b1d-11eb-20da-5550295a62e2
z

# ╔═╡ 426d8974-6b1d-11eb-2695-e717771e1e1c
z + z

# ╔═╡ 45c1173a-6b1d-11eb-16fa-f5bf93bd3c12
length(methods(+))

# ╔═╡ 545d94da-6b1d-11eb-3504-b92c13736822


# ╔═╡ 60069ffc-6b1d-11eb-23b2-3bdf37d7afc1
Base.:+(x::Dual, y::Real) = Dual(x.value + y, x.deriv)

# ╔═╡ 6e7969fc-6b1d-11eb-18b3-3f5be89f87a8
z + 3

# ╔═╡ 6ffc6644-6b1d-11eb-06ce-c30b7003373f
3 + z

# ╔═╡ 77d0defe-6b1d-11eb-3e0a-0bebbf5efd4c
Base.:+(x::Real, y::Dual) = y + x

# ╔═╡ a54ae120-6b1d-11eb-1596-f14abf3743e4
3 + z

# ╔═╡ a6ac35a2-6b1d-11eb-30f5-99e80935361a
Base.:*(f::Dual, g::Dual) = Dual(f.value * g.value, f.deriv * g.value + g.deriv * f.value)

# ╔═╡ b0e03cc6-6b1d-11eb-0b87-aff94ce6cde5
md"""
$$(f * g)'(a) = f'(a) * g(a) + g'(a) * f(a)$$
"""

# ╔═╡ e3f91218-6b1d-11eb-2b3b-d18c833995ac
z

# ╔═╡ e7b1ccec-6b1d-11eb-22ba-ff374be6b510
z * z

# ╔═╡ e9d04224-6b1d-11eb-1764-2b917908a2b6
Base.:^(z::Dual, n::Integer) = Base.power_by_squaring(z, n)

# ╔═╡ ef3f5a2e-6b1d-11eb-01ba-e786b50e1b8a
z^2

# ╔═╡ 311b4284-6b1e-11eb-0ade-2365544ee664
z

# ╔═╡ 39a3f1d8-6b1e-11eb-2e54-0937cd01220d
z * 2

# ╔═╡ 549b07ce-6b1e-11eb-169c-83c47e38dba4
function autodiff(f, a)
	x = Dual(a, one(a))   # one means the multiplicative identity element
	
	result = f(x)
	
	return result.deriv
end

# ╔═╡ 8351c5b2-6b1e-11eb-2fe7-790ee4438c38
a = 3

# ╔═╡ 7e92019c-6b1e-11eb-054a-f129b8b50e42
xx = Dual(a, 1)

# ╔═╡ b1cdddae-6b1e-11eb-32a0-fd2465fecf53


# ╔═╡ ca5abcae-6b1e-11eb-1eb9-1f1c1aa3f312
2a + 2

# ╔═╡ ed33d86c-6b1e-11eb-3a9d-dbc7e06774e9
one(3)

# ╔═╡ efe39cac-6b1e-11eb-3f6e-23bba7b7a5fc
one(3.1)

# ╔═╡ 0d3bee4a-6b1f-11eb-2d8f-3baa0ef67a20
md"""
$\sin(a + \epsilon b) = \sin(a) + \epsilon \sin'(a) * b$
"""

# ╔═╡ 313c33c4-6b1f-11eb-3b7e-41984348e2da
Base.sin(f::Dual) = Dual(sin(f.value), cos(f.value) * f.deriv)

# ╔═╡ 6195ea2e-6b1f-11eb-2ed7-01168f0796a7
`ChainRules.jl`

# ╔═╡ 783dff0a-6b1f-11eb-32ae-432d05f589ec
autodiff(x -> sin(x^2), 1)

# ╔═╡ 81dda736-6b1f-11eb-2458-b10b45f7ce1a
cos(1) * 2 * 1

# ╔═╡ a7c7d3d6-6b1f-11eb-3520-f7f2933d5796
@tree sin(x^2)

# ╔═╡ cd47fb4a-6b1f-11eb-07f1-1dc651ce9782
p = x^2

# ╔═╡ d61f3fb2-6b1f-11eb-1ecd-dfeb82b20bc3


# ╔═╡ 3dbcb924-6b20-11eb-0dbc-2110d50cc8c0
@variables w, t

# ╔═╡ 47b89218-6b20-11eb-1372-593544251a42
g(x, y) = x^2 + y^2 - 2

# ╔═╡ aeaad72e-6b20-11eb-2a7f-23d16abb675c
typeof(w.val)

# ╔═╡ b9766b8c-6b20-11eb-1336-6b9075307454
w * w

# ╔═╡ e9c8bd22-6b1f-11eb-2feb-cdd8857784fa
md"""
## Using types for dispatch
"""

# ╔═╡ cc6feaf6-6b20-11eb-0a61-b39c948e0455
newton_step(f, f′, x) = x - f(x) / f′(x)

# ╔═╡ d77c2806-6b20-11eb-293a-c1760bb2687d
finite_diff(f, x, h=0.01) = (f(x+h) - f(x-h)) / (2h)

# ╔═╡ 4f8e6cdc-6b21-11eb-305d-23384cb9c6fd


# ╔═╡ e75cfe12-6b20-11eb-3ace-85645b0dda64


# ╔═╡ 13faa406-6b21-11eb-2d65-f597d5ca7909


# ╔═╡ 73db9aec-6b21-11eb-081c-333c02dfb048
abstract type DiffAlg end

# ╔═╡ 4a8a70e6-6b21-11eb-1302-d3f927830e27
struct FiniteDiffAlg <: DiffAlg
	h::Float64
end

# ╔═╡ 6a6ebdb6-6b21-11eb-0324-9590784caab4
struct ForwardDiffAlg <: DiffAlg
end

# ╔═╡ 7e0e6e86-6b21-11eb-06d8-65e3f1daa27e
function newton_step(f, x, diff_algorithm::DiffAlg)
	deriv = diff_algorithm(f, x)
	return x - f(x) / deriv
end

# ╔═╡ bd6bc45c-6b21-11eb-235d-e7cb5be7135c
newton_step(x -> x^2, 1.0, ForwardDiffAlg())

# ╔═╡ f536d156-6b21-11eb-32b8-8147902a17a9
md"""
Need to make this object callable:
"""

# ╔═╡ fdc8593c-6b21-11eb-0812-b740799bbf95
(alg::ForwardDiffAlg)(f, x) = ForwardDiff.derivative(f, x)

# ╔═╡ 27df56f0-6b22-11eb-3b30-7df4d79467a3
newton_step(x -> x^2, 1.0, ForwardDiffAlg())

# ╔═╡ 2c9ff210-6b22-11eb-0ea1-859286cbf6ef


# ╔═╡ 60f8889e-6b22-11eb-3f00-a5f0568f4853


# ╔═╡ 668375da-6b22-11eb-327e-7b393c61420a
md"""
Big picture:

Write an algorithm like Newton's method with a method to specify different algorithms.



"""

# ╔═╡ 6a8d4d1e-6b23-11eb-2d82-77fe0ed030a8
@which sin(3.1)

# ╔═╡ 8823c39c-6b23-11eb-3049-1b068652c13e
@which sin(3 + 4im)

# ╔═╡ be202576-6b23-11eb-2a91-6d7e1330036c
mysin(x) = x - x^3 / factorial(3) + x^5 / factorial(5)

# ╔═╡ d28b405e-6b23-11eb-366c-c553a0e890ee
autodiff(mysin, 1.0)

# ╔═╡ 279a6b30-6b24-11eb-0eed-c17378882977


# ╔═╡ a00762fc-6b1e-11eb-15dc-bdc62a00ada8
f(xx)

# ╔═╡ b66f9816-6b1e-11eb-20f2-2768f883ef6a
f(a)

# ╔═╡ d6b037de-6b1e-11eb-1612-1f7c37d94465
autodiff(f, 3)

# ╔═╡ 3fead4a2-6b24-11eb-2c05-abfea19073df
function ff(v) 
	v_new = copy(v)
	
	v_new .= v .* v
	
	return v_new
	
end

# ╔═╡ 688453fc-6b24-11eb-1574-254d3d81a44b
v = [i for i in 1:1000]

# ╔═╡ 7e9bbc52-6b24-11eb-18a7-f9e4072c7679
ff(v)

# ╔═╡ 848b099c-6b24-11eb-133e-ab29822330eb
with_terminal() do
	@time ff(v)
end

# ╔═╡ bd581e72-6b24-11eb-13d4-17225c447655
typeof(v)

# ╔═╡ cd87c3d8-6b24-11eb-25f0-b16743ddda74
ww = [1, "hello", 3.5, 6]

# ╔═╡ e8b0afc6-6b24-11eb-3053-c11d9881bffd
www = reduce(vcat, [copy(ww) for i in 1:250])

# ╔═╡ 07882622-6b25-11eb-3260-71019125f136
length(www)

# ╔═╡ 0a3c3886-6b25-11eb-0fea-dd694572f869
g(x) = x^2

# ╔═╡ 50d4f6ca-6b20-11eb-22fb-4f23ef537940
ex = g(w, t)

# ╔═╡ 7caff376-6b20-11eb-03da-f9476c79d4dc
ex2 = ModelingToolkit.value(ex);

# ╔═╡ 88e221b4-6b20-11eb-0ee6-6fec23b3d405
Text(ex2.args)

# ╔═╡ 122d2480-6b25-11eb-229e-0370be5033fc
ff(www)

# ╔═╡ 1d387882-6b25-11eb-24b9-2b458d406e48
with_terminal() do
	@time ff(www)
end

# ╔═╡ 38acda64-6b24-11eb-21a7-6d8d95e9da28
f(x) = sin(x)

# ╔═╡ 98eb2b7a-6b1e-11eb-1853-01257b5f3c43
f(x) = x^2 + x + x

# ╔═╡ Cell order:
# ╟─76876386-6b1a-11eb-1038-974c75817b46
# ╟─9a5e166a-6b1a-11eb-3e79-c90c8c86b312
# ╟─eee63e1a-6b1a-11eb-3566-43dd0ee7ba40
# ╠═f2629bd8-6b1a-11eb-3aef-37d264641bd1
# ╠═f9e00f9e-6b1a-11eb-272d-cf3787db796b
# ╠═fd22d768-6b1a-11eb-0d33-77830e60952e
# ╠═35260e00-6b1b-11eb-133d-f74b247b289f
# ╟─391551c4-6b1b-11eb-1c91-bb491a24b056
# ╟─61083ff4-6b1b-11eb-38e9-85721910addc
# ╟─7b0098f2-6b1b-11eb-151b-1981d4b8bba1
# ╟─b0737e08-6b1b-11eb-26e1-87348f0949aa
# ╟─c603da7e-6b1b-11eb-288a-098526cf1d31
# ╟─9a9e0d8e-6b1b-11eb-22f6-2d80328a81b0
# ╠═094fa5d8-6b1c-11eb-10b2-3b9d703c3412
# ╠═34b8bd0e-6b1c-11eb-326b-47f1462c6f4e
# ╠═28f51046-6b1c-11eb-2f58-77f44bdb6146
# ╠═3f42684c-6b1c-11eb-272d-396ff9b9cb73
# ╟─7bd03f14-6b1c-11eb-06f3-2bc07ae05d66
# ╠═455878ac-6b1c-11eb-3adf-add5f4afcd84
# ╠═091f3a3c-6b1d-11eb-13dd-8bb44ae3e6a4
# ╠═0a211ec8-6b1d-11eb-3abc-9bb0a96f1e0b
# ╠═0fd33f2c-6b1d-11eb-1426-07abd3999e0a
# ╠═191a8b44-6b1d-11eb-0805-8bd851504811
# ╟─5a68c3f0-6b1c-11eb-0a36-33a9f12539be
# ╠═4092d352-6b1d-11eb-20da-5550295a62e2
# ╠═426d8974-6b1d-11eb-2695-e717771e1e1c
# ╠═45c1173a-6b1d-11eb-16fa-f5bf93bd3c12
# ╠═545d94da-6b1d-11eb-3504-b92c13736822
# ╠═60069ffc-6b1d-11eb-23b2-3bdf37d7afc1
# ╠═6e7969fc-6b1d-11eb-18b3-3f5be89f87a8
# ╠═6ffc6644-6b1d-11eb-06ce-c30b7003373f
# ╠═77d0defe-6b1d-11eb-3e0a-0bebbf5efd4c
# ╠═a54ae120-6b1d-11eb-1596-f14abf3743e4
# ╠═a6ac35a2-6b1d-11eb-30f5-99e80935361a
# ╠═b0e03cc6-6b1d-11eb-0b87-aff94ce6cde5
# ╠═e3f91218-6b1d-11eb-2b3b-d18c833995ac
# ╠═e7b1ccec-6b1d-11eb-22ba-ff374be6b510
# ╠═e9d04224-6b1d-11eb-1764-2b917908a2b6
# ╠═ef3f5a2e-6b1d-11eb-01ba-e786b50e1b8a
# ╠═311b4284-6b1e-11eb-0ade-2365544ee664
# ╠═39a3f1d8-6b1e-11eb-2e54-0937cd01220d
# ╠═549b07ce-6b1e-11eb-169c-83c47e38dba4
# ╠═8351c5b2-6b1e-11eb-2fe7-790ee4438c38
# ╠═7e92019c-6b1e-11eb-054a-f129b8b50e42
# ╠═98eb2b7a-6b1e-11eb-1853-01257b5f3c43
# ╠═a00762fc-6b1e-11eb-15dc-bdc62a00ada8
# ╠═b1cdddae-6b1e-11eb-32a0-fd2465fecf53
# ╠═b66f9816-6b1e-11eb-20f2-2768f883ef6a
# ╠═ca5abcae-6b1e-11eb-1eb9-1f1c1aa3f312
# ╠═d6b037de-6b1e-11eb-1612-1f7c37d94465
# ╠═ed33d86c-6b1e-11eb-3a9d-dbc7e06774e9
# ╠═efe39cac-6b1e-11eb-3f6e-23bba7b7a5fc
# ╠═0a02bfb2-6b1f-11eb-11f0-ef09f0118de2
# ╠═0d3bee4a-6b1f-11eb-2d8f-3baa0ef67a20
# ╠═313c33c4-6b1f-11eb-3b7e-41984348e2da
# ╠═6195ea2e-6b1f-11eb-2ed7-01168f0796a7
# ╠═783dff0a-6b1f-11eb-32ae-432d05f589ec
# ╠═81dda736-6b1f-11eb-2458-b10b45f7ce1a
# ╠═93196008-6b1f-11eb-058a-79fca3e576a7
# ╠═a7c7d3d6-6b1f-11eb-3520-f7f2933d5796
# ╠═cd47fb4a-6b1f-11eb-07f1-1dc651ce9782
# ╠═d61f3fb2-6b1f-11eb-1ecd-dfeb82b20bc3
# ╠═ef88c752-6b1f-11eb-1572-11c8caa573d1
# ╠═3dbcb924-6b20-11eb-0dbc-2110d50cc8c0
# ╠═47b89218-6b20-11eb-1372-593544251a42
# ╠═50d4f6ca-6b20-11eb-22fb-4f23ef537940
# ╠═7caff376-6b20-11eb-03da-f9476c79d4dc
# ╠═88e221b4-6b20-11eb-0ee6-6fec23b3d405
# ╠═aeaad72e-6b20-11eb-2a7f-23d16abb675c
# ╠═b9766b8c-6b20-11eb-1336-6b9075307454
# ╠═e9c8bd22-6b1f-11eb-2feb-cdd8857784fa
# ╠═cc6feaf6-6b20-11eb-0a61-b39c948e0455
# ╠═d77c2806-6b20-11eb-293a-c1760bb2687d
# ╠═4f8e6cdc-6b21-11eb-305d-23384cb9c6fd
# ╠═e75cfe12-6b20-11eb-3ace-85645b0dda64
# ╠═13faa406-6b21-11eb-2d65-f597d5ca7909
# ╠═73db9aec-6b21-11eb-081c-333c02dfb048
# ╠═4a8a70e6-6b21-11eb-1302-d3f927830e27
# ╠═6a6ebdb6-6b21-11eb-0324-9590784caab4
# ╠═7e0e6e86-6b21-11eb-06d8-65e3f1daa27e
# ╠═bd6bc45c-6b21-11eb-235d-e7cb5be7135c
# ╟─f536d156-6b21-11eb-32b8-8147902a17a9
# ╠═fdc8593c-6b21-11eb-0812-b740799bbf95
# ╠═27df56f0-6b22-11eb-3b30-7df4d79467a3
# ╠═2c9ff210-6b22-11eb-0ea1-859286cbf6ef
# ╠═60f8889e-6b22-11eb-3f00-a5f0568f4853
# ╟─668375da-6b22-11eb-327e-7b393c61420a
# ╠═6a8d4d1e-6b23-11eb-2d82-77fe0ed030a8
# ╠═8823c39c-6b23-11eb-3049-1b068652c13e
# ╠═be202576-6b23-11eb-2a91-6d7e1330036c
# ╠═d28b405e-6b23-11eb-366c-c553a0e890ee
# ╠═279a6b30-6b24-11eb-0eed-c17378882977
# ╠═38acda64-6b24-11eb-21a7-6d8d95e9da28
# ╠═494f6d6e-6b24-11eb-11ce-efda4259cf0a
# ╠═3fead4a2-6b24-11eb-2c05-abfea19073df
# ╠═688453fc-6b24-11eb-1574-254d3d81a44b
# ╠═7e9bbc52-6b24-11eb-18a7-f9e4072c7679
# ╠═848b099c-6b24-11eb-133e-ab29822330eb
# ╠═bd581e72-6b24-11eb-13d4-17225c447655
# ╠═cd87c3d8-6b24-11eb-25f0-b16743ddda74
# ╠═e8b0afc6-6b24-11eb-3053-c11d9881bffd
# ╠═07882622-6b25-11eb-3260-71019125f136
# ╠═0a3c3886-6b25-11eb-0fea-dd694572f869
# ╟─122d2480-6b25-11eb-229e-0370be5033fc
# ╠═1d387882-6b25-11eb-24b9-2b458d406e48
