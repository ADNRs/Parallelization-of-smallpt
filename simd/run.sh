postfix="simd"

for type in float double
do
    echo ">>>>>" $type "<<<<<"
    echo
    for spp in 20 200 2000 20000
    do
        echo --- ${spp}spp ---
        g++ -O3 -D`echo ${type} | tr [a-z] [A-Z]` -fopenmp smallpt_${postfix}.cpp -o smallpt_${postfix} || exit 1
        time ./smallpt_${postfix} $spp 2>/dev/null || exit 1
        mv image.ppm ${type}_${spp}spp.ppm
        echo
    done
done

rm smallpt_${postfix}
