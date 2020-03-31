# Model

You can use either blendedmvs-trained dataset or use dtu-trained model. They are provided in [release](https://github.com/kwea123/CasMVSNet_pl/releases)

## Some notes
During evaluation, since some scenes have more than 800 views, you might need limit the number of refernce views by setting `--max_ref_views` to a proper number (500 by default) as the program requires huge RAM which grows with the image size and the number of reference views. I tested with 32GB RAM, and it is almost all consumed using default settings, if you have more RAM, feel free to set this number larger.

# Quantitative evaluation

[No point cloud evaluation currently](https://github.com/YoYo000/BlendedMVS/issues/4).

# Qualitative evaluation

### 5b08286b2775267d5b0634ba
![teaser](../../assets/5b08286b2775267d5b0634ba.png)

Some interesting and beautiful point clouds will be provided in [release](https://github.com/kwea123/CasMVSNet_pl/releases).
