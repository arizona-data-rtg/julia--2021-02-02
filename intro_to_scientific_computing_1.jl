### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ ad6d5e6c-659f-11eb-39a7-b1ad065e3928
using PlutoUI

# ╔═╡ 5e2eba2e-65a1-11eb-1c3e-0d07187fe1d7
using Plots

# ╔═╡ 7e6fc2b6-659b-11eb-18d7-7138f9b0beb5
md"""
# Intro to scientific computing in Julia

## David P. Sanders
"""

# ╔═╡ bff4c162-659b-11eb-3f96-714b34b36bcc
md"""
## Root finding
"""

# ╔═╡ e958f520-659b-11eb-1634-a10f3412b5a2
md"""
Solve equation

$$f(x) = 0$$

Solve using Newton's method:

$$x_{n+1} = x_n - f(x_n) / f'(x_n)$$
"""

# ╔═╡ 0631499a-659c-11eb-3252-4b2e054630b0
x0 = 1.0

# ╔═╡ 0f1edeaa-659c-11eb-171e-71d1278c4060
x0

# ╔═╡ 12763f94-659c-11eb-17b6-07c176518645
typeof(x0)

# ╔═╡ 3088810e-659c-11eb-3718-e9afe685711f
typeof(1)

# ╔═╡ 38a660cc-659c-11eb-1b48-a1b4cb1d0b38
Float64(1)

# ╔═╡ 3bc8d3a2-659c-11eb-3dcb-cb2691b6a928
md"""
## Implement Newton
"""

# ╔═╡ 405de95e-659c-11eb-1ee7-39bdeff2ac7f
x₀ = 0   # \_0<TAB>

# ╔═╡ 4f3464e2-659c-11eb-3937-13a1a8c7e091
⛄ = 5

# ╔═╡ 760ce256-659c-11eb-06f1-311c5890a2eb
f(x) = x^2 - 2   # short-style function definition

# ╔═╡ 7aea1d16-659c-11eb-2f31-e5bdc3e2af44
f_deriv(x) = 2x

# ╔═╡ e6acff64-659c-11eb-2612-957be6de4d77
function newton_step(f, f_deriv, x0)   # long style function definiti0on
	x1 = x0 - f(x0) / f_deriv(x0)  
	return x1
end

# ╔═╡ 46f61ffe-659d-11eb-2a33-87bbeae41f3d
md"""
Note that we have already seen 3 different syntaxes for defining functions!
Use anonymous functions to define little functions to pass as arguments into other functions
"""

# ╔═╡ b50c4b1c-659d-11eb-1169-31c583b958ee
md"""
## Derivatives
"""

# ╔═╡ b2bb560e-659e-11eb-2f88-eba9e30ea29e
md"""
For numerical derivative, need a parameter: step size.
"""

# ╔═╡ f6eebe20-659d-11eb-0322-79d74d855fe3
deriv(f, x, h=0.01) = (f(x + h) - f(x - h)) / (2h)  # default argument h=0.01

# ╔═╡ bed0a200-659e-11eb-1ba7-599ccfa41933


# ╔═╡ 2d75851a-659f-11eb-36d8-97cf9912e577
@which sin(5.1)

# ╔═╡ 1a6c59ac-659e-11eb-092a-317f8dfa9e04
deriv(f) = x -> deriv(f, x)

# ╔═╡ d1a71094-659e-11eb-3135-cd037c0074be
deriv

# ╔═╡ 0fde7c76-659f-11eb-3b78-31728ee38cdc
methods(deriv)

# ╔═╡ 49a2f2ee-659e-11eb-09b0-9f12dba07024
newton_step(f, x0) = newton_step(f, deriv(f), x0)  # another method

# ╔═╡ 2ff76ae2-659d-11eb-1421-072849542a56
newton_step(x -> x^3, x -> 3x^2, 1.0)    # anonymous function definition x -> f(x)

# ╔═╡ ff08d724-659f-11eb-3868-73fb7a923c29
function newton(f, n, x0)
	x = x0
	
	for i in 1:n
		x = newton_step(f, x)
	end
	
	return x
end

# ╔═╡ 19b65664-65a0-11eb-1883-91e74d3b0f5d
1:10

# ╔═╡ 1d3d4a72-65a0-11eb-2ded-439860f54320
typeof(1:10)

# ╔═╡ 21bf6e18-65a0-11eb-0224-e91341bb5be3
collect(1:10)

# ╔═╡ 57455c8c-65a0-11eb-03dd-73b86415ef1c
newton(x -> x^2 - 2, 10, 1.0)

# ╔═╡ 6ca1bff8-65a0-11eb-046b-03bc03fd2076
sqrt(2)

# ╔═╡ a717abca-65a0-11eb-2015-b74a6d141fa4
function iterate(f, method, n, x0)
	x = x0
	
	xs = [x]  # vector with one entry
	
	for i in 1:n
		x = method(f, x)
		push!(xs, x)    # convention: ! signifies that the function modifies its first argument
	end
	
	return xs
end

# ╔═╡ f933d87a-65a0-11eb-2f50-1b989d54009e
v = [1, 2, 3]

# ╔═╡ 36efed0e-65a1-11eb-3981-2f3ba0648c1a
typeof(v)

# ╔═╡ 42b7a008-65a1-11eb-0617-59b881ca0f72
v[2]

# ╔═╡ 4bcdf744-65a1-11eb-393f-992e41a779a0
push!(v, 10)

# ╔═╡ 57b9010e-65a1-11eb-10e2-f1d319de6762
v + v

# ╔═╡ 398565a0-65a2-11eb-2659-6b3e9a653e98
setprecision(10000)  # BigFloat precision

# ╔═╡ 5fdac3bc-65a2-11eb-3c31-cf4bab241afc


# ╔═╡ b41ba5cc-65a2-11eb-307e-09c80934fe62
md"""
## Type annotations
"""

# ╔═╡ bbb45a68-65a2-11eb-32df-bfbb685d7dc1
md"""
Type annotations never help with speed. They are only for **dispatch**: choosing which version of a function to call, based on the types of the arguments
"""

# ╔═╡ c39e68ea-65a2-11eb-2dc5-33a92ba8b79e
ff(x) = 2x

# ╔═╡ d28a5ef4-65a2-11eb-07c0-fbd886d380b0
ff(x::Int) = 10x

# ╔═╡ ce4ccba6-65a2-11eb-0110-0717262067ca
ff(3)

# ╔═╡ e465f8d6-65a2-11eb-0538-731bd963921e
ff(3.0)

# ╔═╡ f9f12d60-65a2-11eb-0430-e13c273146cf
@which 3 + 4

# ╔═╡ 18e93280-65a3-11eb-30bd-6bfdc2cf9998
@which 3 + 4.5

# ╔═╡ 3429fd18-65a3-11eb-2b86-65ffbce4a85e
methods(+)

# ╔═╡ 4299a04a-65a3-11eb-2ed4-b78f5577cffd
f(x::Float64, y::Float64) = x^2 + y^2

# ╔═╡ 557a3fca-659c-11eb-1ec0-59bfc82729dd
x1 = x0 - f(x0) / f_deriv(x0)   # \prime<TAB>

# ╔═╡ 0465e174-659d-11eb-1040-fb1fbed32240
newton_step(f, f_deriv, 1.0)

# ╔═╡ 0eba163a-659e-11eb-3190-a323c3acbb6a
deriv(f, 1)

# ╔═╡ c33e96b2-659e-11eb-156f-51d6990d4670
deriv(f, 1, 0.001)

# ╔═╡ 6de92c22-659e-11eb-0d09-8998736ea0ae
newton_step(f, 0.1)

# ╔═╡ 96c943fc-659e-11eb-3dc2-a56a5b541395
f(1)

# ╔═╡ 8a827944-659f-11eb-0a35-65abb56fd294
@code_typed f(1)

# ╔═╡ 9eb1a0d6-659f-11eb-3bca-f56e6c976ee8
with_terminal() do
	@code_native f(1)
end

# ╔═╡ be75c1ac-659f-11eb-1b6a-3707c11df36f
@code_typed f(1.5)

# ╔═╡ c9f134b4-659f-11eb-1316-43ad9e6aaabc
with_terminal() do
	@code_native f(1.5)
end

# ╔═╡ cd56fd9a-65a0-11eb-05fb-05f5f578ddc4
xs = iterate(f, newton_step, 10, 0.1)

# ╔═╡ 777882bc-65a1-11eb-3614-95624f51a7e5
plot(xs, m=:o)

# ╔═╡ df1f9356-65a1-11eb-16b7-c1414e75ce69
xs[end]

# ╔═╡ 237fabf8-65a2-11eb-2b3f-971d7395787c
xs2 = iterate(f, newton_step, 10, big(10.0))

# ╔═╡ c8dfc3ae-65a1-11eb-13c9-d5afc6aef7d4
deltas = xs2 .- xs2[end]   # elementwise /  broadcast / vectorised operations

# ╔═╡ faa44e82-65a1-11eb-254e-7fb14299172a
plot(abs.(replace(deltas, 0=>NaN)), m=:o, yscale=:log10)

# ╔═╡ 53e91a30-65a3-11eb-0cb6-0b33e6700680
f(1, 2)

# ╔═╡ 6dec0302-65a3-11eb-118d-c30da8489ddd
M = [1 2
	 3 4]

# ╔═╡ 76c7d8ac-65a3-11eb-0b1d-0fa036e4c60c
M^2  # matrix multiplication M * M

# ╔═╡ 7e2ac532-65a3-11eb-1ad3-21cac506059e
M .^ 2

# ╔═╡ 8677ab38-65a3-11eb-1f9a-b9e1172545cc
f(M, M)

# ╔═╡ a2490570-65a3-11eb-3f39-3d4da11d4dad
fff(x, y) = x'*x + y'*y

# ╔═╡ b0910978-65a3-11eb-31f8-97902833aef2
fff(1, 2)

# ╔═╡ c3f12700-65a3-11eb-0db8-9dcfcc6f4641
1'

# ╔═╡ c6856670-65a3-11eb-3094-955e13db2650
[1, 2]' * [1, 2]

# ╔═╡ e3052966-65a3-11eb-1ed6-bdb72fdf08ac
a = [1 2]

# ╔═╡ e96381ae-65a3-11eb-14b4-a5a4871fc663
b = reshape([1, 2], 2, 1)

# ╔═╡ f8615988-65a3-11eb-3de5-67d95b190e2d
a * b

# ╔═╡ 10f7ffee-65a4-11eb-009d-6523c94c2f18
[1, 2]' * [1, 2]

# ╔═╡ 1e7620d6-65a4-11eb-1f5e-c7401b8d4c46
[1, 2]'

# ╔═╡ 2ae174e2-65a4-11eb-00d6-81583284418b
[1, 2]

# ╔═╡ 31c768de-65a4-11eb-2418-957714a93dd6
[1 2]

# ╔═╡ 3bb2fa5c-65a4-11eb-0cac-b52488547a04
[1, 2] * [1, 2]

# ╔═╡ 43440e76-65a4-11eb-237a-47effe65a956


# ╔═╡ Cell order:
# ╟─7e6fc2b6-659b-11eb-18d7-7138f9b0beb5
# ╠═ad6d5e6c-659f-11eb-39a7-b1ad065e3928
# ╟─bff4c162-659b-11eb-3f96-714b34b36bcc
# ╟─e958f520-659b-11eb-1634-a10f3412b5a2
# ╠═0631499a-659c-11eb-3252-4b2e054630b0
# ╠═0f1edeaa-659c-11eb-171e-71d1278c4060
# ╠═12763f94-659c-11eb-17b6-07c176518645
# ╠═3088810e-659c-11eb-3718-e9afe685711f
# ╠═38a660cc-659c-11eb-1b48-a1b4cb1d0b38
# ╟─3bc8d3a2-659c-11eb-3dcb-cb2691b6a928
# ╠═405de95e-659c-11eb-1ee7-39bdeff2ac7f
# ╠═4f3464e2-659c-11eb-3937-13a1a8c7e091
# ╠═760ce256-659c-11eb-06f1-311c5890a2eb
# ╠═7aea1d16-659c-11eb-2f31-e5bdc3e2af44
# ╠═557a3fca-659c-11eb-1ec0-59bfc82729dd
# ╠═e6acff64-659c-11eb-2612-957be6de4d77
# ╠═0465e174-659d-11eb-1040-fb1fbed32240
# ╠═2ff76ae2-659d-11eb-1421-072849542a56
# ╟─46f61ffe-659d-11eb-2a33-87bbeae41f3d
# ╟─b50c4b1c-659d-11eb-1169-31c583b958ee
# ╟─b2bb560e-659e-11eb-2f88-eba9e30ea29e
# ╠═f6eebe20-659d-11eb-0322-79d74d855fe3
# ╠═bed0a200-659e-11eb-1ba7-599ccfa41933
# ╠═0eba163a-659e-11eb-3190-a323c3acbb6a
# ╠═c33e96b2-659e-11eb-156f-51d6990d4670
# ╠═d1a71094-659e-11eb-3135-cd037c0074be
# ╠═0fde7c76-659f-11eb-3b78-31728ee38cdc
# ╠═2d75851a-659f-11eb-36d8-97cf9912e577
# ╠═1a6c59ac-659e-11eb-092a-317f8dfa9e04
# ╠═49a2f2ee-659e-11eb-09b0-9f12dba07024
# ╠═6de92c22-659e-11eb-0d09-8998736ea0ae
# ╠═96c943fc-659e-11eb-3dc2-a56a5b541395
# ╠═8a827944-659f-11eb-0a35-65abb56fd294
# ╠═9eb1a0d6-659f-11eb-3bca-f56e6c976ee8
# ╠═be75c1ac-659f-11eb-1b6a-3707c11df36f
# ╠═c9f134b4-659f-11eb-1316-43ad9e6aaabc
# ╠═ff08d724-659f-11eb-3868-73fb7a923c29
# ╠═19b65664-65a0-11eb-1883-91e74d3b0f5d
# ╠═1d3d4a72-65a0-11eb-2ded-439860f54320
# ╠═21bf6e18-65a0-11eb-0224-e91341bb5be3
# ╠═57455c8c-65a0-11eb-03dd-73b86415ef1c
# ╠═6ca1bff8-65a0-11eb-046b-03bc03fd2076
# ╠═a717abca-65a0-11eb-2015-b74a6d141fa4
# ╠═cd56fd9a-65a0-11eb-05fb-05f5f578ddc4
# ╠═f933d87a-65a0-11eb-2f50-1b989d54009e
# ╠═36efed0e-65a1-11eb-3981-2f3ba0648c1a
# ╠═42b7a008-65a1-11eb-0617-59b881ca0f72
# ╠═4bcdf744-65a1-11eb-393f-992e41a779a0
# ╠═57b9010e-65a1-11eb-10e2-f1d319de6762
# ╠═5e2eba2e-65a1-11eb-1c3e-0d07187fe1d7
# ╠═777882bc-65a1-11eb-3614-95624f51a7e5
# ╠═df1f9356-65a1-11eb-16b7-c1414e75ce69
# ╠═c8dfc3ae-65a1-11eb-13c9-d5afc6aef7d4
# ╠═faa44e82-65a1-11eb-254e-7fb14299172a
# ╠═237fabf8-65a2-11eb-2b3f-971d7395787c
# ╠═398565a0-65a2-11eb-2659-6b3e9a653e98
# ╠═5fdac3bc-65a2-11eb-3c31-cf4bab241afc
# ╟─b41ba5cc-65a2-11eb-307e-09c80934fe62
# ╠═bbb45a68-65a2-11eb-32df-bfbb685d7dc1
# ╠═c39e68ea-65a2-11eb-2dc5-33a92ba8b79e
# ╠═ce4ccba6-65a2-11eb-0110-0717262067ca
# ╠═d28a5ef4-65a2-11eb-07c0-fbd886d380b0
# ╠═e465f8d6-65a2-11eb-0538-731bd963921e
# ╠═f9f12d60-65a2-11eb-0430-e13c273146cf
# ╠═18e93280-65a3-11eb-30bd-6bfdc2cf9998
# ╟─3429fd18-65a3-11eb-2b86-65ffbce4a85e
# ╠═4299a04a-65a3-11eb-2ed4-b78f5577cffd
# ╠═53e91a30-65a3-11eb-0cb6-0b33e6700680
# ╠═6dec0302-65a3-11eb-118d-c30da8489ddd
# ╠═76c7d8ac-65a3-11eb-0b1d-0fa036e4c60c
# ╠═7e2ac532-65a3-11eb-1ad3-21cac506059e
# ╠═8677ab38-65a3-11eb-1f9a-b9e1172545cc
# ╠═a2490570-65a3-11eb-3f39-3d4da11d4dad
# ╠═b0910978-65a3-11eb-31f8-97902833aef2
# ╠═c3f12700-65a3-11eb-0db8-9dcfcc6f4641
# ╠═c6856670-65a3-11eb-3094-955e13db2650
# ╠═e3052966-65a3-11eb-1ed6-bdb72fdf08ac
# ╠═e96381ae-65a3-11eb-14b4-a5a4871fc663
# ╠═f8615988-65a3-11eb-3de5-67d95b190e2d
# ╠═10f7ffee-65a4-11eb-009d-6523c94c2f18
# ╠═1e7620d6-65a4-11eb-1f5e-c7401b8d4c46
# ╠═2ae174e2-65a4-11eb-00d6-81583284418b
# ╠═31c768de-65a4-11eb-2418-957714a93dd6
# ╠═3bb2fa5c-65a4-11eb-0cac-b52488547a04
# ╠═43440e76-65a4-11eb-237a-47effe65a956
