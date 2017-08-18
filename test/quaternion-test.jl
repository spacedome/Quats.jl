
@testset "quat basic tests" begin

    @testset "type conversion" for T in (Int32, Int64, Float32, Float64)
        @test real(quat(T)) == T
        @test complex(quat(T)) == complex(T)
    end

    @test real(quat(1.0)) == 1.0
    @test real(quat(1.0+2.0im)) == 1.0
    @test imag(quat(1.0+2.0im)) == 2.0
    @test jmag(1.0+2.0im+3.0jm+4.0km) == 3.0
    @test kmag(1.0+2.0im+3.0jm+4.0km) == 4.0
    @test quat(vec([0.0 2.0 3.0 4.0])) == 2im + 3jm + 4km
    @test isinteger(quat(1,0,0,0))
    @test !isinteger(quat(1.1, 0.0, 0.0, 0.0))
    @test !isinteger(jm)
    @test isfinite(quat(1.0, 10.0, 100.0, 1000.0))
    @test !isfinite(quat(1.0, Inf, Inf, Inf))
    @test isnan(quat(NaN, NaN, NaN, NaN))
    @test !isnan(1 + im + jm + km)
    @test isinf(quat(1.0, Inf, Inf, Inf))
    @test !isinf(quat(1.0, 10.0, 100.0, 1000.0))
    @test iszero(quat(0.0, 0.0, 0.0, 0.0))
    @test iszero(quat(0, 0, 0, 0))
    @test !iszero(im + jm + km)
    @test quat(jm+km) == jm+km
    @test complex(1+im+jm+km) == 1+im
    @test vec(1+2im+3jm+4km) == [1; 2; 3; 4]

    @test conj(1+2im+3jm+4km) == 1-2im-3jm-4km

    @test quat(false,true,false,false) == im
    @test quat(false,false,true,false) == jm
    @test quat(false,false,false,true) == km
    @test quat(1,0,0,0) == 1
    @test quat(0,1,0,0) == im
    @test quat(0,0,1,0) == jm
    @test quat(0,0,0,1) == km
    @test quat(1.0,0.0,0.0,0.0) == 1
    @test quat(0.0,1.0,0.0,0.0) == im
    @test quat(0.0,0.0,1.0,0.0) == jm
    @test quat(0.0,0.0,0.0,1.0) == km
    @test quat(-1,0,0,0) == -1
    @test quat(0,-1,0,0) == -im
    @test quat(0,0,-1,0) == -jm
    @test quat(0,0,0,-1) == -km
    @test quat(-1.0,0.0,0.0,0.0) == -1
    @test quat(0.0,-1.0,0.0,0.0) == -im
    @test quat(0.0,0.0,-1.0,0.0) == -jm
    @test quat(0.0,0.0,0.0,-1.0) == -km
    @test im*jm ==  km == -jm*im 
    @test jm*im == -km == -im*jm
    @test jm*km ==  im == -km*jm
    @test km*jm == -im == -jm*km
    @test km*im ==  jm == -im*km
    @test im*km == -jm == -km*im
    @test im*im == jm*jm == km*km == -1 

end

# These mirror tests from Complex.jl
@testset "arithmetic" begin
    @testset for T in (Float16, Float32, Float64, BigFloat)
        t = true
        f = false
        u = quat(T(+1.0), T(+1.0), T(+1.0), T(+1.0))

        @testset "add and subtract" begin

            @test isequal(T(+0.0) + im + jm + km, quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(-0.0) + im + jm + km, quat(T(-0.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(+0.0) - im - jm - km, quat(T(+0.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(-0.0) - im - jm - km, quat(T(-0.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(+1.0) + im + jm + km, quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(-1.0) + im + jm + km, quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(T(+1.0) - im - jm - km, quat(T(+1.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(T(-1.0) - im - jm - km, quat(T(-1.0),T(-1.0),T(-1.0),T(-1.0)))
            @test isequal(jm + T(+0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm + T(-0.0), quat(T(-0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm - T(+0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(jm - T(-0.0), quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0)))
            @test isequal(km + T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km + T(-0.0), quat(T(-0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km - T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(km - T(-0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0)))
            @test isequal(im + jm + km + T(+1.0), quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km + T(-1.0), quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km - T(+1.0), quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test isequal(im + jm + km - T(-1.0), quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0)))
            @test T(f) + im + jm + km == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test T(t) + im + jm + km == quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0))
            @test T(f) - im - jm - km == quat(T(+0.0),T(-1.0),T(-1.0),T(-1.0))
            @test T(t) - im - jm - km == quat(T(+1.0),T(-1.0),T(-1.0),T(-1.0))
            @test im + jm + km + T(f) == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km + T(t) == quat(T(+1.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km - T(f) == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            @test im + jm + km - T(t) == quat(T(-1.0),T(+1.0),T(+1.0),T(+1.0))
        end

        @testset "multiply" begin
            oq  = quat(T(+1.0),T(+0.0),T(+0.0),T(+0.0))
            iq  = quat(T(+0.0),T(+1.0),T(+0.0),T(+0.0))
            jq  = quat(T(+0.0),T(+0.0),T(+1.0),T(+0.0))
            kq  = quat(T(+0.0),T(+0.0),T(+0.0),T(+1.0))
            oqn = quat(T(-1.0),T(-0.0),T(-0.0),T(-0.0))
            iqn = quat(T(-0.0),T(-1.0),T(-0.0),T(-0.0))
            jqn = quat(T(-0.0),T(-0.0),T(-1.0),T(-0.0))
            jkn = quat(T(-0.0),T(-0.0),T(-0.0),T(-1.0))
            @test isequal(T(+1.0) * u,  u)
            @test isequal(u * T(+1.0),  u)
            @test isequal(T(-1.0) * u, -u)
            @test isequal(u * T(-1.0), -u)
            @test isequal(T(+0.0) * u, quat(T(+0.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(u * T(+0.0), quat(T(+0.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(T(-0.0) * u, quat(T(-0.0),T(-0.0),T(-0.0),T(-0.0)))
            @test isequal(u * T(-0.0), quat(T(-0.0),T(-0.0),T(-0.0),T(-0.0)))
            @test isequal(oq * oq, oq)
            @test isequal(iq * iq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(jq * jq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test isequal(kq * kq, quat(T(-1.0),T(+0.0),T(+0.0),T(+0.0)))
            @test iq*jq ==  kq == -jq*iq 
            @test jq*iq == -kq == -iq*jq
            @test jq*kq ==  iq == -kq*jq
            @test kq*jq == -iq == -jq*kq
            @test kq*iq ==  jq == -iq*kq
            @test iq*kq == -jq == -kq*iq
        end

        @testset "divide" begin
            @test u/T(+1.0) == u
        end
    end

end

# @testset "exp(q)" begin
    

@testset "Matrix tests" begin
    
    Q = [1+2im+3jm+4jm -im+km; -jm+km -5+6im-7jm-8km]

    @test qmatrix(cmatrix(1+2im+3jm+4km))[1,1] == 1+2im+3jm+4km
    @test qmatrix(cmatrix(Q)) == Q
end
    
# end