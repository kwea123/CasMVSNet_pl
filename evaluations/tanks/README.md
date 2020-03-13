# Quantitative evaluation

1.  Prepare `.ply` files from depth fusion, and `.log` files from [preprocessed files](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view).
2.  Register at [tanks and temples](https://www.tanksandtemples.org/).
3.  Follow their submission guidelines.

## Result

### Intermediate

|   | Mean   | Family | Francis | Horse  | Lighthouse | M60    | Panther | Playground | Train |
|---|--------|--------|---------|--------|------------|--------|---------|------------|-------|
|[Original](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)| 56.42  | 76.36  | 58.45   | 46.20  | 55.53	  | 56.11  | 54.02   | 58.17	  | 46.56 |
|This repo| 55.09 | 76.40 |	52.83 |	49.08 |	49.72 |	56.24 |	51.99 |	53.87 |	50.63

### Advanced

|   | Mean   | Auditorium |	Ballroom | Courtroom | Museum	| Palace | Temple | 
|---|--------|------------|----------|-----------|--------|--------|---------|
|[Original](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)| 31.12  | 19.81  | 38.46   | 29.10  | 43.87	  | 27.36  | 28.11   |
|This repo| 29.63 | 20.57 |	35.10 |	28.88 |	36.05 |	26.84 |	30.31 |

[Detailed results](https://www.tanksandtemples.org/details/829/)

# Qualitative evaluation

A video of Train (click to link to YouTube):
[![teaser](../../assets/train.gif)](https://youtu.be/5NkF6Xbe-1o)

The point clouds are provided in [release](https://github.com/kwea123/CasMVSNet_pl/releases).
