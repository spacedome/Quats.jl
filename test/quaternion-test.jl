
@testset "quat basic tests" begin

    @testset "type conversion" for T in (Int64, Float64)
        @test real(quat(T)) == T
        @test complex(quat(T)) == complex(T)
        @test real(quat(1.0)) == 1.0
        @test real(quat(1.0+2.0im)) == 1.0
        @test imag(quat(1.0+2.0im)) == 2.0
        @test jmag(1.0+2.0im+3.0jm+4.0km) == 3.0
        @test kmag(1.0+2.0im+3.0jm+4.0km) == 4.0
    end

    @test quat(false,true,false,false) == im
    @test quat(false,false,true,false) == jm
    @test quat(false,false,false,true) == km
    @test quat(0,1,0,0) == im
    @test quat(0,0,1,0) == jm
    @test quat(0,0,0,1) == km
    @test quat(0.0,1.0,0.0,0.0) == im
    @test quat(0.0,0.0,1.0,0.0) == jm
    @test quat(0.0,0.0,0.0,1.0) == km
    @test im*jm ==  km == -jm*im 
    @test jm*im == -km == -im*jm
    @test jm*km ==  im == -km*jm
    @test km*jm == -im == -jm*km
    @test km*im ==  jm == -im*km
    @test im*km == -jm == -km*im
    @test im*im == jm*jm == km*km == -1 

end

@testset "arithmetic" begin
    @testset for T in (Float16, Float32, Float64, BigFloat)
        t = true
        f = false
        u = quat(T(+1.0), T(+1.0), T(+1.0), T(+1.0))

        @testset "add and subtract" begin
            @test T(+0.0) + im + jm + km == quat(T(+0.0),T(+1.0),T(+1.0),T(+1.0))
            # @test isequal(T(+0.0) + im, Complex(T(+0.0), T(+1.0)))
            # @test isequal(T(-0.0) + im, Complex(T(-0.0), T(+1.0)))
            # @test isequal(T(+0.0) - im, Complex(T(+0.0), T(-1.0)))
            # @test isequal(T(-0.0) - im, Complex(T(-0.0), T(-1.0)))
            # @test isequal(T(+1.0) + im, Complex(T(+1.0), T(+1.0)))
            # @test isequal(T(-1.0) + im, Complex(T(-1.0), T(+1.0)))
            # @test isequal(T(+1.0) - im, Complex(T(+1.0), T(-1.0)))
            # @test isequal(T(-1.0) - im, Complex(T(-1.0), T(-1.0)))
            # @test isequal(im + T(+0.0), Complex(T(+0.0), T(+1.0)))
            # @test isequal(im + T(-0.0), Complex(T(-0.0), T(+1.0)))
            # @test isequal(im - T(+0.0), Complex(T(+0.0), T(+1.0)))
            # @test isequal(im - T(-0.0), Complex(T(+0.0), T(+1.0)))
            # @test isequal(im + T(+1.0), Complex(T(+1.0), T(+1.0)))
            # @test isequal(im + T(-1.0), Complex(T(-1.0), T(+1.0)))
            # @test isequal(im - T(+1.0), Complex(T(-1.0), T(+1.0)))
            # @test isequal(im - T(-1.0), Complex(T(+1.0), T(+1.0)))
            # @test isequal(T(f) + im, Complex(T(+0.0), T(+1.0)))
            # @test isequal(T(t) + im, Complex(T(+1.0), T(+1.0)))
            # @test isequal(T(f) - im, Complex(T(+0.0), T(-1.0)))
            # @test isequal(T(t) - im, Complex(T(+1.0), T(-1.0)))
            # @test isequal(im + T(f), Complex(T(+0.0), T(+1.0)))
            # @test isequal(im + T(t), Complex(T(+1.0), T(+1.0)))
            # @test isequal(im - T(f), Complex(T(+0.0), T(+1.0)))
            # @test isequal(im - T(t), Complex(T(-1.0), T(+1.0)))
        end

        @testset "multiply" begin
            @test T(+1.0) * u == u * T(+1.0) == u
        end

        @testset "divide" begin
            @test u/T(+1.0) == u
        end
    end

end
