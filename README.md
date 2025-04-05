# map raster-tiles from openDTM data

The fantastic https://www.opendem.info provides us with a unified terrain grid over germany,
collected from open data of each federal state.

The resolution is pretty good (1 pixel ~= 1m²) and this leads to a couple of challenges.

This is my approach to resample them to web-mercator at the highest possible resolution and
also calculate the normal map to allow lighting/shading in a WebGL viewer.


```shell
python src/cli.py reproject -z 13 -r 3600
```

    zoom  1: 13,831,678 x 19,995,929 - 13,831,678 x 19,995,929
    zoom  2: 16,817,554 x 9,564,058  - 16,817,554 x 9,564,058
    zoom  3:  3,802,618 x 3,299,783  -  3,802,618 x 3,299,783
    zoom  4:  1,894,705 x 1,727,497  -  1,894,705 x 1,727,497
    zoom  5:    823,746 x 799,082    -    972,309 x 968,352
    zoom  6:    393,226 x 392,205    -    454,879 x 453,497
    zoom  7:    192,100 x 191,699    -    219,735 x 219,117
    zoom  8:     94,239 x 94,053     -    111,781 x 111,473
    zoom  9:     47,132 x 47,038     -     56,370 x 56,216
    zoom 10:     23,453 x 23,407     -     28,305 x 28,227
    zoom 11:     11,727 x 11,704     -     14,152 x 14,113
    zoom 12:      5,863 x 5,852      -      7,083 x 7,064
    zoom 13:      2,933 x 2,927      -      3,540 x 3,530
    zoom 14:      1,466 x 1,463      -      1,770 x 1,765
    zoom 15:        733 x 732        -        884 x 882
    zoom 16:        366 x 366        -        442 x 441
    zoom 17:        183 x 183        -        221 x 220
    zoom 18:         91 x 91         -        110 x 110
    zoom 19:         45 x 45         -         55 x 55
    zoom 20:         22 x 22         -         27 x 27
