### Quaternion type heavily based off of Base lib Complex.jl

"""
    Quaternion{T<:Real} <: Number
Quaternion number type with real and imaginary parts of type `T`.
`Quaternion256` and `Quaternion128` are aliases for
`Quaternion{Float64}` and `Quaternion{Float32}` respectively.
"""
struct Quaternion{T<:Real} <: Number
    re::T
    im::T
    jm::T
    km::T
end

### Constructors
Quaternion(a::Real, b::Real, c::Real, d::Real) = Quaternion(promote(a,b,c,d)...)
Quaternion(x::Real) = Quaternion(x, zero(x), zero(x), zero(x))
Quaternion(z::Complex) = Quaternion(real(z), imag(z), zero(real(z)), zero(real(z)))
Quaternion(q::Quaternion) = q
### need this one for BigInt/BigFloat convert
Quaternion{T}(q::Quaternion) where {T <: Real} = Quaternion(T(q.re), T(q.im), T(q.jm), T(q.jm))

### For representation as q = z + c*j for complex z and c
Quaternion(z1::Complex, z2::Complex) = Quaternion(real(z1), imag(z1), real(z2), imag(z2))

# Doesn't seem like a good design, fix this ?
Quaternion(v::Vector{<:Real}) = Quaternion(v[1], v[2], v[3], v[4])

"""
    jm
Quaternion imaginary unit.
"""
const jm = Quaternion(false,false,true,false)
"""
    km
Quaternion imaginary unit.
"""
const km = Quaternion(false,false,false,true)

const QuaternionF16 = Quaternion{Float16}
const QuaternionF32 = Quaternion{Float32}
const QuaternionF64 = Quaternion{Float64}
const QuatF16 = Quaternion{Float16}
const QuatF32 = Quaternion{Float32}
const QuatF64 = Quaternion{Float64}

convert(::Type{Quaternion{T}}, x::Real) where {T<:Real} = Quaternion{T}(x,0,0,0)
convert(::Type{Quaternion{T}}, z::Complex) where {T<:Real} = Quaternion{T}(real(z),imag(z),0,0)
convert(::Type{Quaternion{T}}, q::Quaternion) where {T<:Real} = Quaternion{T}(q.re, q.im, q.jm, q.km)

convert(::Type{Quaternion}, q::Quaternion) = q
convert(::Type{Quaternion}, z::Complex) = Quaternion(z)
convert(::Type{Quaternion}, x::Real) = Quaternion(x)


promote_rule(::Type{Quaternion{T}}, ::Type{S}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Complex{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}


widen(::Type{Quaternion{T}}) where {T} = Quaternion{widen(T)}

bswap(q::Quaternion) = Quaternion(bswap(q.re), bswap(q.im), bswap(q.jm), bswap(q.km))

### Redefine/overload methods for Complex
real(q::Quaternion) = q.re
imag(q::Quaternion) = q.im

"""
    jmag(q)
Return the j-imaginary part of the quaternion number `q`.
```jldoctest
julia> jmag(1 + 3im + 2jm + 5km)
2
```
"""
jmag(q::Quaternion) = q.jm

"""
    kmag(q)
Return the k-imaginary part of the quaternion number `q`.
```jldoctest
julia> kmag(1 + 3im + 2jm + 5km)
5
```
"""
kmag(q::Quaternion) = q.km

real(::Type{Quaternion{T}}) where {T<:Real} = T

isreal(q::Quaternion) = iszero(q.im) && iszero(q.jm) && iszero(q.km)
isinteger(q::Quaternion) = isreal(q) && isinteger(q.re)
isfinite(q::Quaternion) = isfinite(q.re) && isfinite(q.im) && isfinite(q.jm) && isfinite(q.km)
isnan(q::Quaternion) = isnan(q.re) || isnan(q.im) || isnan(q.jm) || isnan(q.km)
isinf(q::Quaternion) = isinf(q.re) || isinf(q.im) || isinf(q.jm) || isinf(q.km)
iszero(q::Quaternion) = iszero(q.re) && iszero(q.im) && iszero(q.jm) && iszero(q.km)
iscomplex(q::Quaternion) = iszero(q.jm) && iszero(q.km)
isone(q::Quaternion) = isreal(q) && isone(q.re)

zero(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{T}(zero(T), zero(T), zero(T), zero(T))

"""
    quat(r, [i], [j], [k])
Convert real numbers or arrays to quaternion. `i`, `j`, `k`, default to zero.
Essentially just a more convienient way to call the constructor Quaternion(...).
"""
quat(a::Real, b::Real, c::Real, d::Real) = Quaternion(a, b, c, d)
quat(x::Real) = Quaternion(x)
quat(z::Complex) = Quaternion(z)
quat(q::Quaternion) = q
quat(z1::Complex, z2::Complex) = Quaternion(z1, z2)
quat(v::Vector{<:Real}) = Quaternion(v)

complex(::Type{Quaternion{T}}) where {T<:Real} = Complex{T}
complex(q::Quaternion) = complex(q.re, q.im)


"""
    quat(T::Type)
Returns an appropriate type which can represent a value of type `T` as a quaternion.
Equivalent to `typeof(quat(zero(T)))`.
```jldoctest
julia> quat(Quaternion{Int})
Quaternion{Int64}
julia> quat(Int)
Quaternion{Int64}
```
"""
quat(::Type{T}) where {T<:Real} = Quaternion{T}
quat(::Type{Complex{T}}) where {T<:Real} = Quaternion{T}
quat(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{T}

vec(q::Quaternion) = vcat(q.re, q.im, q.jm, q.km)

function show(io::IO, q::Quaternion)
    r, i, j, k = vec(q)
    compact = get(io, :compact, false)
    show(io, r)
    for (x, xm) in [(i, "im"), (j, "jm"), (k, "km")]
        if signbit(x) && !isnan(x)
            x = -x
            print(io, compact ? "-" : " - ")
        else
            print(io, compact ? "+" : " + ")
        end
        show(io, x)
        if !(isa(x,Integer) && !isa(x,Bool) || isa(x,AbstractFloat) && isfinite(x))
            print(io, "*")
        end
        print(io, xm)
    end
end

function show(io::IO, q::Quaternion{Bool})
    if q == im
        print(io, "im")
    elseif q == jm
        print(io, "jm")
    elseif q == km
        print(io, "km")
    else
        print(io, "Quaternion($(q.re),$(q.im),$(q.jm),$(q.km))")
    end
end

function show(io::IO, ::MIME"text/html", q::Quaternion)
    r, i, j, k = vec(q)
    compact = get(io, :compact, false)
    show(io, r)
    for (x, xm) in [(i, "i"), (j, "j"), (k, "k")]
        if signbit(x) && !isnan(x)
            x = -x
            print(io, compact ? "-" : " - ")
        else
            print(io, compact ? "+" : " + ")
        end
        show(io, x)
        if !(isa(x,Integer) && !isa(x,Bool) || isa(x,AbstractFloat) && isfinite(x))
            print(io, "*")
        end
        print(io, "<b><i>" * xm * "</i></b>")
    end
end

function show(io::IO, ::MIME"text/html", q::Quaternion{Bool})
    if q == im
        print(io, "<b><i>i</i></b>")
    elseif q == jm
        print(io, "<b><i>j</i></b>")
    elseif q == km
        print(io, "<b><i>k</i></b>")
    else
        print(io, "Quaternion($(q.re),$(q.im),$(q.jm),$(q.km))")
    end
end

function read(s::IO, ::Type{Quaternion{T}}) where {T}
    r = read(s,T)
    i = read(s,T)
    j = read(s,T)
    k = read(s,T)
    Quaternion{T}(r,i,j,k)
end
function write(s::IO, q::Quaternion)
    write(s,q.re,q.im,q.jm,q.km)
end

==(q::Quaternion, w::Quaternion) = q.re == w.re && q.im == w.im && q.jm == w.jm && q.km == w.km
==(q::Quaternion, z::Complex) = iscomplex(q) && q.im == imag(z) && q.re == real(z)
==(q::Quaternion, x::Real) = isreal(q) && q.re == x

isequal(q::Quaternion, w::Quaternion) = isequal(q.re,w.re) & isequal(q.im,w.im) &
                                        isequal(q.jm,w.jm) & isequal(q.km,w.km)

#TODO: hash

conj(q::Quaternion) = Quaternion(q.re,-q.im,-q.jm,-q.km)
abs(q::Quaternion)  = sqrt(abs2(q))
abs2(q::Quaternion) = q.re*q.re + q.im*q.im + q.jm*q.jm + q.km*q.km
inv(q::Quaternion)  = conj(q)/abs2(q)
inv(q::Quaternion{<:Integer}) = inv(float(q))

-(q::Quaternion) = Quaternion(-q.re, -q.im, -q.jm, -q.km)
+(q::Quaternion, w::Quaternion) = Quaternion(q.re + w.re, q.im + w.im,
                                             q.jm + w.jm, q.km + w.km)
-(q::Quaternion, w::Quaternion) = Quaternion(q.re - w.re, q.im - w.im,
                                             q.jm - w.jm, q.km - w.km)
*(q::Quaternion, w::Quaternion) = Quaternion(q.re*w.re - q.im*w.im - q.jm*w.jm - q.km*w.km,
                                             q.re*w.im + q.im*w.re + q.jm*w.km - q.km*w.jm,
                                             q.re*w.jm - q.im*w.km + q.jm*w.re + q.km*w.im,
                                             q.re*w.km + q.im*w.jm - q.jm*w.im + q.km*w.re)


+(x::Real, q::Quaternion) = Quaternion(x + q.re, q.im, q.jm, q.km)
+(q::Quaternion, x::Real) = Quaternion(q.re + x, q.im, q.jm, q.km)
+(z::Complex, q::Quaternion) = Quaternion(real(z) + q.re, imag(z) + q.im, q.jm, q.km)
+(q::Quaternion, z::Complex) = Quaternion(q.re + real(z), q.im + imag(z), q.jm, q.km)
function -(x::Real, q::Quaternion)
    # we don't want the default type for -(Bool)
    re = x - q.re
    Quaternion(re, - oftype(re, q.im), - oftype(re, q.jm), - oftype(re, q.km))
end
function -(z::Complex, q::Quaternion)
    # we don't want the default type for -(Bool)
    re = real(z) - q.re
    Quaternion(re, imag(z) - q.im, - oftype(re, q.jm), - oftype(re, q.km))
end
-(q::Quaternion, x::Real) = Quaternion(q.re - x, q.im, q.jm, q.km)
-(q::Quaternion, z::Complex) = Quaternion(q.re - real(z), q.im - imag(z), q.jm, q.km)
*(x::Real, q::Quaternion) = Quaternion(x * q.re, x * q.im, x * q.jm, x * q.km)
*(q::Quaternion, x::Real) = Quaternion(q.re * x, q.im * x, q.jm * x, q.km * x)

*(q::Quaternion, z::Complex) = Quaternion(q.re*z.re - q.im*z.im, q.re*z.im + q.im*z.re,
                                                      q.jm*z.re + q.km*z.im, q.km*z.re - q.jm*z.im)
*(z::Complex, q::Quaternion) = Quaternion(z.re*q.re - z.im*q.im, z.re*q.im + z.im*q.re,
                                                      z.re*q.jm - z.im*q.km, z.re*q.km + z.im*q.jm)

/(a::R, q::S) where {R<:Real,S<:Quaternion} = (T = promote_type(R,S); a*inv(T(q)))
/(q::Quaternion, x::Real) = Quaternion(q.re/x, q.im/x, q.jm/x, q.km/x)
/(q::Quaternion, w::Quaternion) = q * inv(w)


rand(r::AbstractRNG, ::SamplerType{Quaternion{T}}) where {T<:Real} =
    Quaternion(rand(r,T), rand(r,T), rand(r,T), rand(r,T))
"""
When the type argument is quaternion, the values are drawn
from the circularly symmetric quaternion normal distribution.
This is std normal in each component but with variance scaled by 1/4.
"""
randn(r::AbstractRNG, ::Type{Quaternion{T}}) where {T<:AbstractFloat} =
    Quaternion(T(0.5)*randn(r,T), T(0.5)*randn(r,T), T(0.5)*randn(r,T), T(0.5)*randn(r,T))

norm(q::Quaternion{T}, p::Real=2) where {T<:Real} = norm(vec(q), p)
normalize(q::Quaternion{T}, p::Real=2) where {T<:Real} = Quaternion(normalize(vec(q), p))

exp(q::Quaternion{<:Integer}) = exp(float(q))
log(q::Quaternion{<:Integer}) = log(float(q))


### See Neil Dantam - Quaternion Computation - for exp and log
function exp(q::Quaternion{<:AbstractFloat})
    V = norm([q.im q.jm q.km])
    s = if V < 1e-8
        1 - V*V/6 + V*V*V*V/120
    else
        sin(V)/V
    end
    exp(q.re)*(cos(V) + quat(zero(q.re), q.im*s, q.jm*s, q.km*s))
end

function log(q::Quaternion{<:AbstractFloat})
    V = norm([q.im q.jm q.km])
    Q = abs(q)
    ϕ = atan(V, q.re)
    s = if V < 1e-8
        (1+ϕ*ϕ/6 +7*ϕ*ϕ*ϕ*ϕ/360)/Q
    else
        ϕ/V
    end
    quat(log(Q), q.im*s, q.jm*s, q.km*s)
end

### NOTE: sqrt(q^2) == q is NOT true for pure quaternion for this computation
sqrt(q::Quaternion) = exp(0.5*log(q))
^(q::Quaternion, w::Quaternion) = exp(w*log(q))

function round(q::Quaternion, rr::RoundingMode=RoundNearest, ri::RoundingMode=rr; kwargs...)
    Quaternion(round(real(q), rr; kwargs...),
               round(imag(q), ri; kwargs...),
               round(jmag(q), ri; kwargs...),
               round(kmag(q), ri; kwargs...))
end

float(q::Quaternion{<:AbstractFloat}) = q
float(q::Quaternion) = Quaternion(float(q.re), float(q.im), float(q.jm), float(q.km))

big(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{big(T)}
big(q::Quaternion{T}) where {T<:Real} = Quaternion{big(T)}(q)


## Matrix representations of Quaternions

function cmatrix(q::Quaternion)
    [complex( q.re, q.im) complex(q.jm,  q.km);
     complex(-q.jm, q.km) complex(q.re, -q.im)]
end

function cmatrix(Q::Matrix{Quaternion{T}}) where {T}
    [complex.( real.(Q),  imag.(Q)) complex.( jmag.(Q),  kmag.(Q));
     complex.(-jmag.(Q),  kmag.(Q)) complex.( real.(Q), -imag.(Q))]
end

function qmatrix(C::Matrix{Complex{T}}) where {T}
    n, m = size(C)
    quat.(C[1:n÷2, 1:m÷2], C[1:n÷2, m÷2+1:m])
end

## Array operations on complex numbers ##

quat(A::AbstractArray{<:Quaternion}) = A
