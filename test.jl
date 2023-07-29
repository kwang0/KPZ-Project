using PyPlot


plt.clf()
N = 100000
A = rand(Float64, N)
sort!(A)
for i in 1:(N-1)
  A[i] = A[i+1] - A[i]
end
pop!(A)
sort!(A)

binwidth = (A[N-1]-A[1])/ (N/100)
plt.hist(A, bins=LinRange(A[1], A[N-1], N ÷ 100))
ts = LinRange(0,10/N,1001)
f = N * N * binwidth * exp.(-N * ts)

plt.plot(ts, f)