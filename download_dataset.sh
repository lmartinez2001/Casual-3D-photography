
#!/bin/bash

#Here's the list of all possible datasets from the Casual3D collection side by side:
#boat-shed (1.4G)     british-museum (3.7G)    cafe (484M)          church (1.3G)
#clowns (284M)        creepy-attic (1.2G)      forest-rock (1.6G)   gas-works-park (1.5G)
#gravity (220M)       gum-wall (2.3G)          gymnasium (1.2G)     jakobstad-museum (1.2G)
#kerry-park (258M)    kitchen (202M)           library-mobile (1.8G) library (1.5G)
#pike-place (1.5G)    sofa (524M)              troll (284M)         water-tower (1.5G)


BASE_URL="http://visual.cs.ucl.ac.uk/pubs/casual3d/datasets"
DATASET="creepy-attic"

wget "$BASE_URL/$DATASET.zip" && unzip "$DATASET.zip" && rm "$DATASET.zip"
echo "Downloaded and extracted $DATASET"