@echo off

set order=2
set envmap=studio
set mesh=mesh

@echo [STEP 1] Compute envmap sh coefficients ...

python utils/spherical_harmonics.py ^
     --order %order%                ^
     --envmap data/%envmap%.exr     ^
     --out_dir output/

@echo [STEP 2] Compute zonal harmonics ...

python utils/zonal_harmonics.py ^
     --l_max %order%            ^
     --out_dir output/

@echo [STEP 3] Compute precompute spherical harmonics ...

python utils/precompute_radiance_transfer.py ^
     --mesh data/%mesh%.obj                  ^
     --out_dir output/

@echo [STEP 4] Run real-time demo ...

python analytic_sh_area_lights.py ^
     --dim 720                    ^
     --envmap %envmap%            ^
     --model %mesh%               ^
     --prt                        ^
     --max_l 2
