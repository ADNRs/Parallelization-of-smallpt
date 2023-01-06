# remove ppm
for name in openmp simd opencl cuda
do
    cd $name
    rm *.ppm *.txt
    cd ..
done

# warmup
echo ----- warmup 5000spp -----
cd openmp
g++ -O3 -fopenmp smallpt_openmp.cpp -o smallpt_openmp
time ./smallpt_openmp 5000 2>/dev/null
cd ..
echo

# run scripts
for name in openmp simd opencl cuda
do
    echo ----- $name starts -----
    cd $name
    echo
    ./run.sh &>$name.txt
    echo
    cd ..
done
